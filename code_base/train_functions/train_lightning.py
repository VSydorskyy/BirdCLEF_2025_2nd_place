import os
from itertools import chain
from pprint import pprint
from time import time
from typing import Callable, Dict, List, Optional, Union

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from code_base.utils import load_json
from code_base.utils.swa import create_chkp


class LitTrainer(L.LightningModule):
    def __init__(
        self,
        model,
        forward,
        optimizer,
        scheduler,
        scheduler_params: Dict,
    ):
        super().__init__()

        self.model = model
        self._forward = forward
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._scheduler_params = scheduler_params

    def _aggregate_outputs(self, losses, inputs, outputs):
        united = losses
        united.update({"input_" + k: v for k, v in inputs.items()})
        united.update({"output_" + k: v for k, v in outputs.items()})
        return united

    def training_step(self, batch):

        start_time = time()
        losses, inputs, outputs = self._forward(self, batch, epoch=self.current_epoch)
        model_time = time() - start_time

        for k, v in losses.items():
            self.log(
                "train_" + k,
                v,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
                batch_size=inputs["target"].shape[0],
                sync_dist=True,
            )
            self.log(
                "train_avg_" + k,
                v,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=inputs["target"].shape[0],
                sync_dist=True,
            )
        self.log(
            "train_model_time",
            model_time,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            batch_size=1,
            sync_dist=True,
        )
        self.log(
            "train_avg_model_time",
            model_time,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=1,
            sync_dist=True,
        )

        return self._aggregate_outputs(losses, inputs, outputs)

    def validation_step(self, batch, batch_idx):

        start_time = time()
        losses, inputs, outputs = self._forward(self, batch, epoch=self.current_epoch)
        model_time = time() - start_time

        for k, v in losses.items():
            self.log(
                "valid_" + k,
                v,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
                batch_size=inputs["target"].shape[0],
                sync_dist=True,
            )
            self.log(
                "valid_avg_" + k,
                v,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=inputs["target"].shape[0],
                sync_dist=True,
            )
        self.log(
            "valid_model_time",
            model_time,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            batch_size=1,
            sync_dist=True,
        )
        self.log(
            "valid_avg_model_time",
            model_time,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=1,
            sync_dist=True,
        )

        return self._aggregate_outputs(losses, inputs, outputs)

    def configure_optimizers(self):
        if self._scheduler is None:
            return [self._optimizer]
        else:
            scheduler = {"scheduler": self._scheduler}
            scheduler.update(self._scheduler_params)
            return (
                [self._optimizer],
                [scheduler],
            )


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def lightning_training(
    train_df: Optional[pd.DataFrame],
    val_df: Optional[pd.DataFrame],
    exp_name: str,
    fold_id: Optional[int],
    seed: int,
    train_dataset_class: torch.utils.data.Dataset,
    val_dataset_class: Optional[torch.utils.data.Dataset],
    train_dataset_config: dict,
    val_dataset_config: Optional[dict],
    train_dataloader_config: dict,
    val_dataloader_config: Optional[dict],
    nn_model_class: torch.nn.Module,
    nn_model_config: dict,
    optimizer_init: Callable,
    scheduler_init: Callable,
    scheduler_params: dict,
    forward: Union[torch.nn.Module, Callable],
    # It is not really Callable. It just lambda that will init List of callbacks
    # each time. It is just done for safe CV training.
    callbacks: Optional[Callable],
    n_epochs: int,
    main_metric: str,
    metric_mode: str,
    checkpoint_callback_params: dict = {},
    tensorboard_logger_params: dict = {},
    trainer_params: dict = {},
    precision_mode: str = "32-true",
    n_checkpoints_to_save: int = 3,
    log_every_n_steps: int = 100,
    debug: bool = False,
    check_exp_exists: bool = False,
    train_strategy: str = "auto",
    nocall_dataset_class: Optional[torch.utils.data.Dataset] = None,
    nocall_dataset_config: Optional[dict] = None,
    nocall_dataset_df_path: Optional[str] = None,
    device_outside_model: bool = False,
    pretrain_config: Optional[Union[dict, List[dict]]] = None,
    class_weights_path: Optional[str] = None,
    power_value: float = -0.5,
    label_str2int_path: Optional[str] = None,
    selected_birds: Optional[List[str]] = None,
    use_sampler: bool = False,
    sampler_with_replacement: bool = False,
    print_model: bool = False,
):
    if check_exp_exists and os.path.exists(os.path.join(exp_name, "checkpoints")):
        raise RuntimeError(f"Folder {exp_name} already exists!")
    if debug:
        val_df = train_df.copy()
    # Set numpy reproducibility
    np.random.seed(seed)
    # Set device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Training Device : {device}")

    use_class_weights = class_weights_path is not None
    if use_class_weights and class_weights_path.endswith(".json") and label_str2int_path is None:
        raise ValueError("Class weights require label_str2int mapping")
    if use_sampler and not use_class_weights:
        raise ValueError("Sampler requires class weights")

    train_dataset = train_dataset_class(
        df=train_df,
        **train_dataset_config,
    )
    is_concat_dataset = False

    if nocall_dataset_class is not None and nocall_dataset_config is not None and nocall_dataset_df_path is not None:
        nocall_df = pd.read_csv(nocall_dataset_df_path)
        nocall_dataset = nocall_dataset_class(
            df=nocall_df,
            **nocall_dataset_config,
        )
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, nocall_dataset])
        is_concat_dataset = True

    if use_class_weights:
        if class_weights_path == "sqrt":
            labels = pd.Series(train_dataset.targets)
            class_weights = (labels.value_counts() / labels.value_counts().sum()) ** power_value
            class_weights = class_weights.to_dict()
        elif class_weights_path == "balanced":
            labels = pd.Series(train_dataset.targets)
            class_weights = 1.0 / labels.value_counts()
            class_weights = class_weights.to_dict()
        else:
            class_weights = load_json(class_weights_path)
        print("Using next class weights:")
        pprint(class_weights)
    if use_class_weights and use_sampler:
        if is_concat_dataset:
            sample_weights = np.array(
                [class_weights[el] for el in list(chain(*[ds.targets for ds in train_dataset.datasets]))]
            )
        else:
            sample_weights = np.array([class_weights[el] for el in train_dataset.targets])
        if hasattr(train_dataset, "dataset_repeat") and train_dataset.dataset_repeat > 1:
            sample_weights = np.concatenate([sample_weights for _ in range(train_dataset.dataset_repeat)])
        assert len(sample_weights) == len(train_dataset)
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, len(sample_weights), replacement=sampler_with_replacement
        )
        print("Sampler Created")
    else:
        sampler = None

    loaders = {
        "train": torch.utils.data.DataLoader(
            train_dataset,
            worker_init_fn=worker_init_fn,
            sampler=sampler,
            **train_dataloader_config,
        ),
    }

    if val_df is not None:
        val_dataset = val_dataset_class(
            df=val_df,
            **val_dataset_config,
        )
        loaders["valid"] = torch.utils.data.DataLoader(
            val_dataset, worker_init_fn=worker_init_fn, **val_dataloader_config
        )
    else:
        print("Skipping validation dataset")

    if device_outside_model:
        model = nn_model_class(**nn_model_config).to(device)
    else:
        model = nn_model_class(device=device, **nn_model_config)

    if print_model:
        print(model)

    if pretrain_config is not None:
        if isinstance(pretrain_config, list):
            current_pretrain_config = pretrain_config[fold_id]
        else:
            current_pretrain_config = pretrain_config
        pretrain_checkpoint = create_chkp(**current_pretrain_config)
        print("Missing keys: ", set(model.state_dict().keys()) - set(pretrain_checkpoint))
        print("Extra keys: ", set(pretrain_checkpoint) - set(model.state_dict().keys()))
        model.load_state_dict(pretrain_checkpoint, strict=False)

    for k in loaders.keys():
        print(f"{k} Loader Len = {len(loaders[k])}")

    optimizer = optimizer_init(model)
    if scheduler_init is not None:
        scheduler = scheduler_init(optimizer, len(loaders["train"]))
    else:
        scheduler = None

    if not isinstance(forward, torch.nn.Module):
        forward = forward()

    if hasattr(forward, "use_weights") and forward.use_weights:
        print("Setting loss weights ...")
        label_str2int = load_json(label_str2int_path)
        class_weights_array = (
            pd.Series({label_str2int[k]: class_weights[k] for k in label_str2int.keys()})
            .sort_index()
            .values.astype(np.float32)
        )
        forward.set_weights(class_weights_array, device)

    if hasattr(forward, "use_slected_indices") and forward.use_slected_indices:
        print("Setting slected_indices ...")
        label_str2int = load_json(label_str2int_path)
        forward.set_selected_indices([label_str2int[el] for el in selected_birds], device)

    lightning_model = LitTrainer(
        model,
        forward=forward,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_params=scheduler_params,
    )

    all_callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(exp_name, "checkpoints"),
            save_top_k=n_checkpoints_to_save,
            mode=metric_mode,
            monitor=main_metric,
            **checkpoint_callback_params,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    if callbacks is not None:
        all_callbacks += callbacks()

    tensorboard_logger = pl_loggers.TensorBoardLogger(
        save_dir=os.path.join(exp_name, "tensorboard"),
        **tensorboard_logger_params,
    )
    trainer = L.Trainer(
        devices=-1,
        precision=precision_mode,
        strategy=train_strategy,
        max_epochs=n_epochs,
        logger=tensorboard_logger,
        log_every_n_steps=log_every_n_steps,
        callbacks=all_callbacks,
        **trainer_params,
    )
    if "valid" in loaders.keys():
        trainer.fit(model=lightning_model, train_dataloaders=loaders["train"], val_dataloaders=loaders["valid"])
    else:
        trainer.fit(model=lightning_model, train_dataloaders=loaders["train"])
