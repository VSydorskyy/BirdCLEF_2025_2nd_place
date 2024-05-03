import json
from itertools import chain
from typing import Optional, Tuple

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from pytorch_toolbelt.utils import all_gather, broadcast_from_master, is_main_process
from ..utils import groupby_np_array, score_numpy, stack_and_max_by_samples


class ROC_AUC_Score(Callback):
    def __init__(
        self,
        metric_name: str = "roc_auc",
        pred_key: str = "logit",
        target_key: str = "target",
        loader_names: Tuple = ("valid"),
        use_sigmoid: bool = True,
        use_timewise_avarage: bool = False,
        aggr_key: Optional[str] = None,
        # pred_long_key: Optional[str] = None,
        verbose: bool = True,
        label_str2int_mapping_path: Optional[str] = None,
        scored_bird_path: Optional[str] = None,
    ):
        self.metric_name = metric_name
        self.loader_names = loader_names

        self.pred_key = pred_key
        self.target_key = target_key

        self.running_preds = []
        self.running_targets = []

        self.use_sigmoid = use_sigmoid
        self.use_timewise_avarage = use_timewise_avarage
        self.verbose = verbose

        if label_str2int_mapping_path is not None and scored_bird_path is not None:
            print("PaddedCMAPScore will be computed on subset of classes")
            label_str2int = json.load(open(label_str2int_mapping_path))
            scored_bird = json.load(open(scored_bird_path))
            self.scored_bird_ids = [label_str2int[el] for el in scored_bird]
        else:
            self.scored_bird_ids = None

        if aggr_key is not None:
            self.aggr_key = aggr_key
            self.running_aggr = []
        else:
            self.aggr_key = None

        # if pred_long_key is not None:
        #     self.pred_long_key = pred_long_key
        #     self.running_preds_long = []
        # else:
        #     self.pred_long_key = None

        self.accums = {
            loader_name: {
                "preds": [],
                # "preds_long": [],
                "targets": [],
                "sample_ids": [],
            }
            for loader_name in loader_names
        }

    def initialize_accums(self, loader_name):
        self.accums[loader_name] = {
            "preds": [],
            # "preds_long": [],
            "targets": [],
            "sample_ids": [],
        }

    def update_accums(self, outputs, loader_name):
        pred = outputs["output_" + self.pred_key].detach()
        if self.use_sigmoid:
            pred = torch.sigmoid(pred)
        if self.use_timewise_avarage:
            pred = pred.max(axis=1)[0]
        pred = pred.cpu().numpy()
        self.accums[loader_name]["preds"].append(pred)

        self.accums[loader_name]["targets"].append(outputs["input_" + self.target_key].detach().cpu().numpy())

        if self.aggr_key is not None:
            self.accums[loader_name]["sample_ids"].append(outputs["input_" + self.aggr_key].detach().cpu().numpy())

        # if self.pred_long_key is not None:
        #     output_long = outputs["output_" + self.pred_long_key]
        #     if self.use_sigmoid:
        #         output_long = torch.sigmoid(output_long)
        #     if self.use_timewise_avarage:
        #         output_long = output_long.max(axis=1)[0]
        #     output_long = output_long.detach().cpu().numpy()
        #     self.accums[loader_name]["preds_long"].append(output_long)

    def compute_roc_auc(self, pl_module, loader_name):
        preds = all_gather(self.accums[loader_name]["preds"])
        targets = all_gather(self.accums[loader_name]["targets"])
        if self.aggr_key is not None:
            sample_ids = all_gather(self.accums[loader_name]["sample_ids"])
        # if self.pred_long_key is not None:
        #     preds_long = all_gather(self.accums[loader_name]["preds_long"])

        if is_main_process():

            preds = np.concatenate(list(chain(*preds)), axis=0)
            targets = np.concatenate(list(chain(*targets)), axis=0)

            if self.scored_bird_ids is not None:
                targets = targets[:, self.scored_bird_ids]
                preds = preds[:, self.scored_bird_ids]

            # if self.pred_long_key is not None:
            #     preds_long = np.concatenate(list(chain(*preds_long)), axis=0)
            #     if self.scored_bird_ids is not None:
            #         preds_long = preds_long[:, self.scored_bird_ids]

            if self.aggr_key is not None:
                sample_ids = np.concatenate(list(chain(*sample_ids)), axis=0)
                targets = groupby_np_array(
                    groupby_f=sample_ids,
                    array_to_group=targets,
                    apply_f=stack_and_max_by_samples,
                )
                preds = groupby_np_array(
                    groupby_f=sample_ids,
                    array_to_group=preds,
                    apply_f=stack_and_max_by_samples,
                )
                # if self.pred_long_key is not None:
                #     preds_long = groupby_np_array(
                #         groupby_f=sample_ids,
                #         array_to_group=preds_long,
                #         apply_f=stack_and_max_by_samples,
                #     )
            # In order to handle sanity check
            if (targets.shape[0] < 100 and self.aggr_key is not None) or (
                targets.shape[0] < 300 and self.aggr_key is None
            ):
                roc_auc = -1
            else:
                roc_auc = score_numpy(y_true=targets, y_pred=preds)

        else:
            roc_auc = None

        roc_auc = broadcast_from_master(roc_auc)

        pl_module.log(
            loader_name + "_" + self.metric_name,
            roc_auc,
        )

    def on_train_epoch_start(self, trainer, pl_module):
        if "train" in self.loader_names:
            self.initialize_accums("train")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if "train" in self.loader_names:
            self.update_accums(outputs, "train")

    def on_train_epoch_end(self, trainer, pl_module):
        if "train" in self.loader_names:
            self.compute_roc_auc(pl_module, "train")
            self.initialize_accums("train")

    def on_validation_epoch_start(self, trainer, pl_module):
        if "valid" in self.loader_names:
            self.initialize_accums("valid")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if "valid" in self.loader_names:
            self.update_accums(outputs, "valid")

    def on_validation_epoch_end(self, trainer, pl_module):
        if "valid" in self.loader_names:
            self.compute_roc_auc(pl_module, "valid")
            self.initialize_accums("valid")
