import os
import subprocess
from os.path import join as pjoin
from shutil import copyfile

import onnx
import torch
from onnxconverter_common import float16
from onnxsim import simplify


class ONNXEnsemble(torch.nn.Module):
    def __init__(
        self,
        model_class,
        configs,
        device="cpu",
        avarage_type="mean",
        weights=None,
        final_activation=None,
        extract_spec_ones=False,
    ):
        if avarage_type not in ["mean", "gaus", "identity"]:
            raise ValueError("avarage_type must be mean, gaus or identity")
        if weights is not None and avarage_type != "mean":
            raise ValueError("avarage_type must be mean if weights is not None")
        if final_activation is not None and final_activation not in [
            "sigmoid",
            "softmax",
        ]:
            raise ValueError(f"final_activation {final_activation} not implemented")
        super().__init__()
        self.models = torch.nn.ModuleList([model_class(**config, device=device) for config in configs])
        self.avarage_type = avarage_type
        if weights is not None:
            assert len(weights) == len(self.models)
            self.register_buffer("weights", torch.FloatTensor(weights))
        else:
            self.weights = None
        self.final_activation = final_activation
        self.extract_spec_ones = extract_spec_ones

    def forward(self, sample):
        if self.weights is None:
            if self.avarage_type == "identity":
                dim = -1
            else:
                dim = 0

            if self.extract_spec_ones:
                spec = self.models[0].extract_spec(sample)
                pred = torch.stack(
                    [one_model(input=None, spec=spec) for one_model in self.models],
                    dim=dim,
                )
            else:
                pred = torch.stack(
                    [one_model(sample) for one_model in self.models],
                    dim=dim,
                )
        else:
            if self.extract_spec_ones:
                spec = self.models[0].extract_spec(sample)
                pred = torch.stack(
                    [one_model(input=None, spec=spec) * weight for one_model, weight in zip(self.models, self.weights)],
                    dim=0,
                )
            else:
                pred = torch.stack(
                    [one_model(sample) * weight for one_model, weight in zip(self.models, self.weights)],
                    dim=0,
                )
        if self.weights is not None:
            pred = pred.sum(dim=0) / self.weights.sum()
        elif self.avarage_type == "mean":
            pred = pred.mean(dim=0)
        elif self.avarage_type == "gaus":
            pred = (pred**2).mean(axis=0) ** 0.5
        if self.final_activation is not None:
            if self.final_activation == "sigmoid":
                pred = torch.sigmoid(pred)
            elif self.final_activation == "softmax":
                pred = torch.softmax(pred, dim=1)
        return pred


def convert_to_onnx(
    model_to_convert,
    sample_input,
    base_path,
    save_not_simplified=False,
    use_fp16=False,
    use_openvino=False,
    opset_version=12,
):
    os.makedirs(base_path)
    torch.onnx.export(
        # model_to_convert.half() if use_fp16 else model_to_convert,
        model_to_convert,
        # sample_input.half() if use_fp16 else sample_input,
        sample_input,
        pjoin(base_path, "model.onnx"),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "input_batch_size"},
            "output": {0: "output_batch_size"},
        },
    )
    # run checks
    onnx_model = onnx.load(pjoin(base_path, "model.onnx"))
    try:
        onnx.checker.check_model(onnx_model)
        # run additional checks and simplify
        model_simp, check = simplify(onnx_model, skip_fuse_bn=True)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, pjoin(base_path, "model_simpl.onnx"))
    except Exception as e:
        print("ONNX model could not be validated, because of:", e)
        onnx.checker.check_model(pjoin(base_path, "model.onnx"))
        copyfile(
            pjoin(base_path, "model.onnx"),
            pjoin(base_path, "model_simpl.onnx"),
        )
    if use_fp16 and not use_openvino:
        print("Converting ONNX to float16")
        onnx_model = onnx.load(pjoin(base_path, "model_simpl.onnx"))
        onnx_model = float16.convert_float_to_float16(onnx_model)
        onnx.save(onnx_model, pjoin(base_path, "model_simpl.onnx"))
        # run checks
        onnx_model = onnx.load(pjoin(base_path, "model_simpl.onnx"))
        onnx.checker.check_model(onnx_model)
    if use_openvino:
        print("Converting ONNX to OpenVINO")
        openvino_postfix = "_openvino"
        if use_fp16:
            openvino_postfix += "_fp16"
        subprocess.call(
            [
                "ovc",
                pjoin(base_path, "model_simpl.onnx"),
                "--output_model",
                pjoin(base_path + openvino_postfix, "model_simpl"),
                "--compress_to_fp16",
                "True" if use_fp16 else "False",
            ]
        )
    if not save_not_simplified:
        os.remove(pjoin(base_path, "model.onnx"))
