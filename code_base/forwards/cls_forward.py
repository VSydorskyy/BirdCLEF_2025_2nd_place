import numpy as np
import torch
import torch.nn as nn

from ..losses.combined_losses import BCEFocal2WayLoss
from ..losses.focal_loss import BCEFocalLossPaper, FocalLoss, FocalLossBCE

EPSILON_FP16 = 1e-5


class MultilabelClsForwardLongShort(nn.Module):
    def __init__(
        self,
        loss_type="baseline",
        framewise_pred_coef=0.5,
        binirize_labels=False,
        use_weights=False,
        class_sum=False,
        use_slected_indices=False,
        batch_aug=None,
        is_output_dict=True,
        use_focal_loss=False,
        use_bce_focal_loss=False,
        use_bce_focal_loss_paper=False,
        config_2way_loss={
            "weights": [1, 1],
            "clipwise_name": "clipwise_logits_long",
            "framewise_name": "framewise_logits_long",
        },
        mixup_alpha=None,
        mixup_inf_norm=False,
        mixup_binarized_tgt=False,
        mixup_equal_data_w=False,
        mixup_div_factor=None,
        unlabeled_masking_prob=None,
        unlabeled_alpha=None,
        unlabeled_inf_norm=True,
        use_masked_loss=False,
    ):
        super().__init__()
        loss_type in [
            "bseline",
            "prob_baseline",
            "logit_clip_and_max_frame",
            "baseline_and_max_frame",
        ]
        if use_slected_indices:
            if not (class_sum or use_weights):
                raise ValueError("Selected indices are supported only with `class_sum` OR `use_weights`")
        if use_focal_loss and use_bce_focal_loss:
            raise ValueError("use_focal_loss and use_bce_focal_loss are mutually exclusive")
        self.class_sum = class_sum
        self.use_weights = use_weights
        self.use_slected_indices = use_slected_indices
        self.loss_type = loss_type
        self.framewise_pred_coef = framewise_pred_coef
        self.binirize_labels = binirize_labels
        self.is_output_dict = is_output_dict
        self.mixup_alpha = mixup_alpha
        self.mixup_inf_norm = mixup_inf_norm
        self.mixup_binarized_tgt = mixup_binarized_tgt
        self.mixup_equal_data_w = mixup_equal_data_w
        self.mixup_div_factor = mixup_div_factor
        self.use_unlabeled_masking = unlabeled_masking_prob is not None
        self.unlabeled_masking_prob = unlabeled_masking_prob
        self.unlabeled_alpha = unlabeled_alpha
        self.unlabeled_inf_norm = unlabeled_inf_norm
        self.use_masked_loss = use_masked_loss
        if not self.is_output_dict:
            if self.loss_type in [
                "baseline_and_max_frame",
                "logit_clip_and_max_frame",
            ]:
                raise ValueError("Output dict is required for `baseline_and_max_frame` or `logit_clip_and_max_frame`")
        if loss_type == "prob_baseline":
            assert not use_focal_loss and not use_bce_focal_loss
            self.loss_f = nn.BCELoss(
                reduction="none"
                if (self.use_weights or self.class_sum or self.use_unlabeled_masking or self.use_masked_loss)
                else "mean"
            )
        elif loss_type == "BCEFocal2WayLoss":
            self.loss_f = BCEFocal2WayLoss(
                **config_2way_loss,
            )
        elif loss_type == "CrossEntropy":
            self.loss_f = nn.CrossEntropyLoss(
                reduction="none"
                if (self.use_weights or self.class_sum or self.use_unlabeled_masking or self.use_masked_loss)
                else "mean"
            )
        else:
            if use_focal_loss:
                print("Using focal loss")
                self.loss_f = FocalLoss(
                    reduction="none"
                    if (self.use_weights or self.class_sum or self.use_unlabeled_masking or self.use_masked_loss)
                    else "mean"
                )
            elif use_bce_focal_loss:
                print("Using BCE focal loss")
                self.loss_f = FocalLossBCE(
                    reduction="none"
                    if (self.use_weights or self.class_sum or self.use_unlabeled_masking or self.use_masked_loss)
                    else "mean"
                )
            elif use_bce_focal_loss_paper:
                print("Using BCE focal loss paper")
                self.loss_f = BCEFocalLossPaper(
                    reduction="none"
                    if (self.use_weights or self.class_sum or self.use_unlabeled_masking or self.use_masked_loss)
                    else "mean"
                )
            else:
                print("Using BCE with logits loss")
                self.loss_f = nn.BCEWithLogitsLoss(
                    reduction="none"
                    if (self.use_weights or self.class_sum or self.use_unlabeled_masking or self.use_masked_loss)
                    else "mean"
                )
        if self.use_weights:
            self.weights = None
        if self.use_slected_indices:
            self.selected_indices = None
        self.batch_aug = batch_aug
        self.batch_aug_device_is_set = False

    def set_weights(self, weights, device=None):
        self.weights = torch.FloatTensor(weights)[None, :]
        if device is not None:
            self.weights = self.weights.to(device)

    def set_selected_indices(self, selected_indices, device=None):
        self.selected_indices = torch.LongTensor(selected_indices)
        if device is not None:
            self.selected_indices = self.selected_indices.to(device)
        if self.use_weights:
            self.weights = self.weights[:, self.selected_indices]

    def mixup(self, data, targets):
        if self.mixup_div_factor is None:
            indices = torch.randperm(data.size(0))
            data2 = data[indices]
            targets2 = targets[indices]

        if self.mixup_div_factor is None:
            if self.mixup_equal_data_w:
                lam = 0.5
            else:
                lam = torch.FloatTensor([np.random.beta(self.mixup_alpha, self.mixup_alpha)]).to(data.device)
            data = data * lam + data2 * (1 - lam)
            targets = targets * lam + targets2 * (1 - lam)
        else:
            assert data.size(0) % self.mixup_div_factor == 0
            datas = data.view(-1, self.mixup_div_factor, *data.shape[1:])
            targetss = targets.view(-1, self.mixup_div_factor, *targets.shape[1:])
            if self.mixup_equal_data_w:
                lam = torch.ones(self.mixup_div_factor)
            else:
                lam = torch.from_numpy(np.random.beta(self.mixup_alpha, self.mixup_alpha, size=(self.mixup_div_factor)))
            lam = lam / lam.sum()
            lam = lam.to(data)

            data = (datas * lam[None, :, None]).sum(dim=1)
            targets = (targetss * lam[None, :, None]).sum(dim=1)

        if self.mixup_inf_norm:
            data = data / (data.abs().max(dim=1, keepdims=True).values + 1e-6)
        if self.mixup_binarized_tgt:
            targets = (targets > 0).float()

        return data, targets

    def forward(self, runner, batch, epoch=None):

        if self.use_weights and self.weights is None:
            raise RuntimeError("Set weights before calling `forward`")

        if self.use_slected_indices and self.selected_indices is None:
            raise RuntimeError("Set selected_indices before calling `forward`")

        if not runner.model.training:
            wave, labels, dfidx, _, _ = batch
            inputs = {
                "target": labels,
                "dfidx": dfidx,
            }
        else:
            if self.use_unlabeled_masking:
                wave, labels, unlabeled_wave = batch
            elif self.use_masked_loss:
                wave, labels, loss_mask = batch
            else:
                wave, labels = batch
            if self.batch_aug is not None:
                if not self.batch_aug_device_is_set:
                    self.batch_aug.to(wave.device)
                    self.batch_aug_device_is_set = True
                wave = self.batch_aug(wave)
                unlabeled_wave = self.batch_aug(unlabeled_wave)
            if self.mixup_alpha is not None:
                wave, labels = self.mixup(wave, labels)
            if self.binirize_labels:
                inputs = {"target": (labels > 0).float()}
            else:
                inputs = {"target": labels}

        if self.use_unlabeled_masking and runner.model.training:
            unlabeled_wave = unlabeled_wave[: wave.size(0)]
            unlabeled_mix_mask = torch.rand(wave.size(0)) < self.unlabeled_masking_prob
            lam = torch.FloatTensor([np.random.beta(self.mixup_alpha, self.mixup_alpha)]).to(wave.device)
            wave[unlabeled_mix_mask] = lam * wave[unlabeled_mix_mask] + (1 - lam) * unlabeled_wave[unlabeled_mix_mask]
            if self.unlabeled_inf_norm:
                wave[unlabeled_mix_mask] = wave[unlabeled_mix_mask] / (
                    wave[unlabeled_mix_mask].abs().max(dim=1, keepdims=True).values + 1e-6
                )

        output = runner.model(wave)

        if self.loss_type == "baseline":
            if self.is_output_dict:
                loss_v = self.loss_f(output["clipwise_logits_long"], labels)
            else:
                loss_v = self.loss_f(output, labels)
        elif self.loss_type == "baseline_and_max_frame":
            loss_v = (
                self.loss_f(output["clipwise_logits_long"], labels) * (1 - self.framewise_pred_coef)
                + self.loss_f(output["framewise_logits_long"].max(1)[0], labels) * self.framewise_pred_coef
            )
        elif self.loss_type == "logit_clip_and_max_frame":
            loss_v = (
                self.loss_f(torch.logit(output["clipwise_pred_long"]), labels) * (1 - self.framewise_pred_coef)
                + self.loss_f(output["framewise_logits_long"].max(1)[0], labels) * self.framewise_pred_coef
            )
        elif self.loss_type == "prob_baseline":
            if self.is_output_dict:
                loss_v = self.loss_f(output["clipwise_pred_long"], labels)
            else:
                loss_v = self.loss_f(output, labels)
        elif self.loss_type == "BCEFocal2WayLoss":
            loss_v = self.loss_f(output, labels)
        elif self.loss_type == "CrossEntropy":
            assert labels.sum(dim=1).max() == 1
            class_indices = torch.argmax(labels, dim=1)
            if self.is_output_dict:
                loss_v = self.loss_f(output["clipwise_logits_long"], class_indices)
            else:
                loss_v = self.loss_f(output, class_indices)

        if self.use_unlabeled_masking or self.use_masked_loss:
            if runner.model.training:
                if self.use_unlabeled_masking:
                    loss_mask = torch.ones_like(loss_v)
                    loss_mask[unlabeled_mix_mask] = (labels[unlabeled_mix_mask] > 0).float()
                loss_v = loss_v * loss_mask
                loss_v = loss_v.sum() / loss_mask.sum()
            else:
                loss_v = loss_v.mean()

        if self.use_slected_indices:
            loss_v = loss_v[:, self.selected_indices]

        if self.use_weights:
            loss_v = (self.weights * loss_v).sum(dim=1).mean()
        elif self.class_sum:
            loss_v = loss_v.sum(dim=1).mean()

        losses = {"loss": loss_v}
        if not self.is_output_dict:
            output = {"logit": output}

        return losses, inputs, output
