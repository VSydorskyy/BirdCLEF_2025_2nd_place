from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import timm
import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram

try:
    from nnAudio.Spectrogram import CQT1992v2
except:
    print("`nnAudioSTFT` was not imported")
try:
    import leaf_audio_pytorch.frontend as frontend
except:
    print("`LEAF` was not imported")

from ..augmentations.spec_augment import CustomFreqMasking, CustomTimeMasking
from ..augmentations.spec_lowerhigh_augment import RandomLowerHighFreq
from ..augmentations.spec_noise_augment import RandomNoise
from ..augmentations.spec_power_augment import RandomSpecPower
from .blocks import (
    AmplitudeToDB,
    AttHead,
    AttHeadSimplified,
    AttnBlock,
    ChannelAgnosticAmplitudeToDB,
    Clasifier,
    NormalizeMelSpec,
    PoolingLayer,
    QuantizableAmplitudeToDB,
    TraceableMelspec,
)


class WaveCNNAttenClasifier(nn.Module):
    def __init__(
        self,
        backbone: Optional[str],
        device: str,
        mel_spec_paramms: Dict[str, Any],
        head_config: Optional[Dict[str, Any]],
        transformer_backbone: bool = False,
        head_type: str = "AttHead",
        top_db: float = 80.0,
        amin: float = 1e-10,
        fixed_amplitude_to_db: bool = False,
        pretrained: bool = True,
        first_conv_name: str = "conv_stem",
        first_conv_stride_overwrite: Optional[Union[int, Tuple[int, int]]] = None,
        exportable: bool = False,
        quantizable: bool = False,
        central_crop_input: Optional[float] = None,
        selected_indices: Optional[List[int]] = None,
        use_sigmoid: bool = False,
        spec_extractor: str = "Melspec",
        add_backbone_config: Optional[Dict[str, Any]] = None,
        deep_supervision_steps: Optional[List[int]] = None,
        permute_backbone_emb: Optional[Tuple[int]] = None,
        no_head_inference_mode: bool = False,
        spec_augment_config: Optional[Dict[str, Any]] = None,
        atten_smoothing_config: Optional[Dict[str, Any]] = None,
        spec_resize: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        add_backbone_config_ = deepcopy(add_backbone_config)
        head_config_ = deepcopy(head_config)
        mel_spec_paramms_ = deepcopy(mel_spec_paramms)

        self.device = device
        self.central_crop_input = central_crop_input
        self.use_selected_indices = selected_indices is not None
        self.quantizable = quantizable
        self.deep_supervision_steps = deep_supervision_steps
        self.transformer_backbone = transformer_backbone
        self.permute_backbone_emb = permute_backbone_emb
        self.no_head_inference_mode = no_head_inference_mode
        self.logmelspec_extractor = self._create_feature_extractor(
            mel_spec_paramms_, exportable, top_db, quantizable, spec_extractor, fixed_amplitude_to_db, amin
        )
        self.spec_resize = spec_resize
        if spec_augment_config is not None:
            self.spec_augment = []
            if "power_aug" in spec_augment_config:
                self.spec_augment.append(RandomSpecPower(**spec_augment_config["power_aug"]))
            if "lower_high_freq" in spec_augment_config:
                self.spec_augment.append(RandomLowerHighFreq(**spec_augment_config["lower_high_freq"]))
            if "freq_mask" in spec_augment_config:
                self.spec_augment.append(CustomFreqMasking(**spec_augment_config["freq_mask"]))
            if "time_mask" in spec_augment_config:
                self.spec_augment.append(CustomTimeMasking(**spec_augment_config["time_mask"]))
            if "white_noise" in spec_augment_config:
                spec_augment_config["white_noise"]["noise_type"] = "white"
                self.spec_augment.append(RandomNoise(**spec_augment_config["white_noise"]))
            if "bandpass_noise" in spec_augment_config:
                spec_augment_config["bandpass_noise"]["noise_type"] = "bandpass"
                self.spec_augment.append(RandomNoise(**spec_augment_config["bandpass_noise"]))
            self.spec_augment = nn.Sequential(*self.spec_augment)
        else:
            self.spec_augment = None
        if backbone is not None:
            add_backbone_config_ = {} if add_backbone_config_ is None else add_backbone_config_
            if self.transformer_backbone:
                head_dropout = add_backbone_config_.pop("head_dropout", 0.0)
                self.backbone = timm.create_model(
                    backbone,
                    pretrained=pretrained,
                    exportable=exportable,
                    in_chans=self._n_specs,
                    **add_backbone_config_,
                )
                self.backbone.head.drop.p = head_dropout
            else:
                self.backbone = timm.create_model(
                    backbone,
                    features_only=True,
                    pretrained=pretrained,
                    exportable=exportable,
                    in_chans=self._n_specs,
                    out_indices=deep_supervision_steps,
                    **add_backbone_config_,
                )
        if first_conv_stride_overwrite is not None:
            if isinstance(first_conv_name, str):
                first_conv_name = [first_conv_name]
            for conv_name in first_conv_name:
                setattr(
                    getattr(self.backbone, conv_name),
                    "stride",
                    first_conv_stride_overwrite,
                )

        self.atten_smoothing_config = atten_smoothing_config
        if atten_smoothing_config is not None:
            self.atten_smoothing = nn.Sequential(
                *[
                    AttnBlock(
                        embed_dim=self.backbone.feature_info.channels()[-1], dropout=atten_smoothing_config["dropout"]
                    )
                    for _ in range(atten_smoothing_config["num_layers"])
                ]
            )
            self.attn_pe = nn.Embedding(atten_smoothing_config["n_steps"], self.backbone.feature_info.channels()[-1])

        if head_config_ is not None and not self.transformer_backbone:
            backbone_channels = self.backbone.feature_info.channels()
            if head_type == "AttHead":
                self.head = AttHead(
                    in_chans=(backbone_channels[-1] if deep_supervision_steps is None else backbone_channels),
                    exportable=exportable,
                    **head_config_,
                )
            elif head_type == "Clasifier":
                self.head = nn.Sequential(
                    PoolingLayer(pool_type=head_config_.pop("pool_type", "AdaptiveAvgPool2d")),
                    Clasifier(
                        nn_embed_size=backbone_channels[-1]
                        if deep_supervision_steps is None
                        else np.sum(backbone_channels),
                        **head_config_,
                    ),
                )
            elif head_type == "AttHeadSimplified":
                assert deep_supervision_steps is None, "Not implemented"
                self.head = AttHeadSimplified(
                    in_chans=backbone_channels[-1],
                    exportable=exportable,
                    **head_config_,
                )
            else:
                raise NotImplementedError(f"{head_type} not implemented")
        else:
            self.head = None
        if self.use_selected_indices:
            self.register_buffer("selected_indices", torch.LongTensor(selected_indices))
        if use_sigmoid or self.transformer_backbone:
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = None
        self.to(self.device)

    def _create_feature_extractor(
        self, mel_spec_paramms, exportable, top_db, quantizable, spec_extractor, fixed_amplitude_to_db, amin
    ):
        if spec_extractor == "Melspec":
            if exportable:
                spec_init = TraceableMelspec
            else:
                spec_init = MelSpectrogram
        elif spec_extractor == "CQT":
            spec_init = CQT1992v2
        elif spec_extractor == "LEAF":
            spec_init = frontend.Leaf
        else:
            raise NotImplementedError(f"{spec_extractor} not implemented")

        if quantizable:
            am2db_init = QuantizableAmplitudeToDB
        elif fixed_amplitude_to_db:
            am2db_init = ChannelAgnosticAmplitudeToDB
        else:
            am2db_init = AmplitudeToDB

        if isinstance(mel_spec_paramms, list):
            self._n_specs = len(mel_spec_paramms)
            return nn.ModuleList(
                [
                    nn.Sequential(
                        spec_init(**mel_spec_paramm, quantizable=True) if quantizable else spec_init(**mel_spec_paramm),
                        am2db_init(top_db=top_db, amin=amin),
                        NormalizeMelSpec(exportable=exportable),
                    )
                    for mel_spec_paramm in mel_spec_paramms
                ]
            )
        else:
            self._n_specs = 1
            if spec_extractor != "LEAF":
                return nn.Sequential(
                    spec_init(**mel_spec_paramms, quantizable=True) if quantizable else spec_init(**mel_spec_paramms),
                    am2db_init(top_db=top_db, amin=amin),
                    NormalizeMelSpec(exportable=exportable),
                )
            else:
                if mel_spec_paramms.pop("normalize", False):
                    return nn.Sequential(
                        spec_init(**mel_spec_paramms, onnx_export=exportable),
                        NormalizeMelSpec(exportable=exportable),
                    )
                elif mel_spec_paramms.pop("normalize_and_db", False):
                    return nn.Sequential(
                        spec_init(**mel_spec_paramms, onnx_export=exportable),
                        am2db_init(top_db=top_db, amin=amin),
                        NormalizeMelSpec(exportable=exportable),
                    )
                else:
                    return spec_init(**mel_spec_paramms, onnx_export=exportable)

    def extract_spec(self, input):
        if self.central_crop_input is not None:
            overall_pad = input.shape[-1] // 2
            input = input[:, overall_pad // 2 : -(overall_pad // 2)]
        if self._n_specs > 1:
            spec = [mel_spec_extractor(input)[:, None] for mel_spec_extractor in self.logmelspec_extractor]
            spec = torch.cat(spec, dim=1)
        else:
            spec = self.logmelspec_extractor(input)[:, None]
        return spec

    def forward(self, input, spec=None, return_spec_feature=False, return_cnn_emb=False):
        if spec is None:
            if self.central_crop_input is not None:
                overall_pad = input.shape[-1] // 2
                input = input[:, overall_pad // 2 : -(overall_pad // 2)]
            if self._n_specs > 1:
                spec = [mel_spec_extractor(input)[:, None] for mel_spec_extractor in self.logmelspec_extractor]
                spec = torch.cat(spec, dim=1)
            else:
                spec = self.logmelspec_extractor(input)[:, None]
        if self.spec_augment is not None and self.training:
            spec = self.spec_augment(spec)
        if self.spec_resize is not None:
            spec = nn.functional.interpolate(spec, size=self.spec_resize, mode="bilinear")
        if not self.quantizable and return_spec_feature:
            return spec
        if self.deep_supervision_steps is not None:
            emb = self.backbone(spec)
        elif self.transformer_backbone:
            emb = self.backbone(spec)
        else:
            emb = self.backbone(spec)[-1]
        if self.permute_backbone_emb is not None:
            emb = emb.permute(*self.permute_backbone_emb)
        if self.atten_smoothing_config is not None:
            B, C, H, W = emb.size()
            emb = emb.reshape(B, C, H * W)
            emb = emb.permute(0, 2, 1)
            pos = torch.arange(0, emb.size(1), dtype=torch.long, device=emb.device)
            pos_emb = self.attn_pe(pos)
            emb = self.atten_smoothing(emb + pos_emb)
        if not self.quantizable and return_cnn_emb:
            return emb
        if self.head is not None:
            logits = self.head(emb)
            if self.use_selected_indices:
                logits = logits[:, self.selected_indices]
            if self.sigmoid is not None:
                logits = self.sigmoid(logits)
            return logits
        else:
            if self.no_head_inference_mode:
                return self.sigmoid(emb)
            else:
                return {
                    "clipwise_logits_long": emb,
                    "clipwise_pred_long": self.sigmoid(emb),
                }
