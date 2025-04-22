import numpy as np
import torch
from tqdm import tqdm

from ..utils import groupby_np_array, stack_and_max_by_samples

# try:
#     import tensorflow as tf
# except ImportError:
#     print("Tensorflow is not installed")


def torch_cat_with_none(tensor_1, tensor_2):
    if tensor_1 is None:
        return tensor_2
    elif tensor_2 is None:
        return tensor_1
    else:
        return torch.cat([tensor_1, tensor_2])


def to_cpu_with_none(tensor):
    if tensor is None:
        return tensor
    else:
        return tensor.cpu()


class BirdsInference:
    def __init__(
        self,
        device,
        verbose=True,
        verbose_tqdm=True,
        use_sigmoid=True,
        avarage_type="mean",
        model_output_key=None,
        use_compiled_fp16=False,
        fake_model_classes=None,
        aggregate_preds=True,
    ):
        self.verbose = verbose
        self.verbose_tqdm = verbose_tqdm
        self.device = device
        self.use_sigmoid = use_sigmoid
        assert avarage_type in ["mean", "vlom", "identity", "gaus"]
        self.avarage_type = avarage_type
        self.model_output_key = model_output_key
        self.use_compiled_fp16 = use_compiled_fp16
        self.fake_model_classes = fake_model_classes
        self.aggregate_preds = aggregate_preds

        self.val_pred = None
        self.val_tgt = None
        self.sample_ids = None

    def _print_v(self, msg):
        if self.verbose:
            print(msg)

    def _tqdm_v(self, generator):
        if self.verbose_tqdm:
            return tqdm(generator)
        else:
            return generator

    def _apply_pred_act(self, input):
        if self.use_sigmoid:
            input = torch.sigmoid(input)

        return input

    def _avarage_preds(self, x):
        if self.avarage_type == "mean":
            return x.mean(0)
        elif self.avarage_type == "vlom":
            x1 = x.prod(axis=0) ** (1.0 / len(x))
            x = x**2
            x = x.mean(axis=0)
            x = x ** (1 / 2)
            return (x + x1) / 2
        elif self.avarage_type == "gaus":
            x = (x**2).mean(axis=0) ** 0.5
            return x
        else:
            return x

    def _model_forward(
        self,
        nn_model,
        wave,
        is_onnx_model=False,
        is_openvino_model=False,
        is_google_model=False,
        google_postprocess=None,
    ):
        if is_onnx_model:
            model_out = torch.from_numpy(
                nn_model.run(
                    None,
                    {"input": wave.numpy()},
                )[0]
            )
            logits = model_out
        elif is_openvino_model:
            logits = torch.from_numpy(nn_model([wave.numpy()])[nn_model.output(0)])
        elif is_google_model:
            if self.device == "cuda":
                with tf.device("/gpu:0"):
                    google_model_output = nn_model.infer_tf(wave.numpy())
                    logits = google_postprocess(google_model_output)
            else:
                google_model_output = nn_model.infer_tf(wave.numpy())
                logits = google_postprocess(google_model_output)
        else:
            logits = nn_model(wave.to(self.device))
            if self.model_output_key is not None:
                logits = logits[self.model_output_key]
            logits = self._apply_pred_act(logits).detach().cpu()

        return logits

    @torch.no_grad()
    def predict_val_loaders(
        self,
        nn_models,
        data_loaders,
        is_onnx_model=False,
        is_openvino_model=False,
        is_google_model=False,
        google_postprocess=None,
    ):
        if isinstance(nn_models[0], list):
            ansamble = True
            n_folds = len(nn_models[0])
            n_models = len(nn_models)
            for i in range(n_models):
                for j in range(n_folds):
                    assert not nn_models[i][j].training
            nn_models = [[nn_models[j][i] for j in range(n_models)] for i in range(n_folds)]
        else:
            ansamble = False
            if not (is_onnx_model or is_openvino_model or is_google_model):
                for i in range(len(nn_models)):
                    assert not nn_models[i].training

        all_preds, all_tgts = [], []

        for data_loader, nn_model in zip(data_loaders, nn_models):
            loader_preds, loader_tgts, loader_agg = [], [], []

            for wave, target, dfidx, start, end in self._tqdm_v(data_loader):
                if ansamble:
                    logits = []
                    for one_nn_model in nn_model:
                        logits_ = self._model_forward(
                            one_nn_model,
                            wave,
                            is_onnx_model=is_onnx_model,
                            is_openvino_model=is_openvino_model,
                            is_google_model=is_google_model,
                            google_postprocess=google_postprocess,
                        )
                        logits.append(logits_)
                    logits = self._avarage_preds(torch.stack(logits, axis=0))
                else:
                    logits = self._model_forward(
                        nn_model,
                        wave,
                        is_onnx_model=is_onnx_model,
                        is_openvino_model=is_openvino_model,
                        is_google_model=is_google_model,
                        google_postprocess=google_postprocess,
                    )

                loader_preds.append(logits.cpu())
                loader_tgts.append(target.numpy())
                loader_agg.append(dfidx.numpy())

            loader_agg = np.concatenate(loader_agg)

            loader_preds = np.concatenate(loader_preds)
            if self.aggregate_preds:
                loader_preds = groupby_np_array(
                    groupby_f=loader_agg,
                    array_to_group=loader_preds,
                    apply_f=stack_and_max_by_samples,
                )
            all_preds.append(loader_preds)

            loader_tgts = np.concatenate(loader_tgts)
            if self.aggregate_preds:
                loader_tgts = groupby_np_array(
                    groupby_f=loader_agg,
                    array_to_group=loader_tgts,
                    apply_f=stack_and_max_by_samples,
                )
            all_tgts.append(loader_tgts)

        return all_tgts, all_preds

    def _model_forward_test(
        self,
        nn_models,
        wave,
        is_onnx_model=False,
        is_openvino_model=False,
        is_google_model=False,
        google_postprocess=None,
    ):
        if not is_google_model:
            wave = wave.to(self.device)
        if self.fake_model_classes is not None:
            print("Fake Prediction")
            models_logits = np.random.uniform(size=(wave.shape[0], self.fake_model_classes))
            if self.use_compiled_fp16:
                models_logits = models_logits.astype(np.float16)
        elif is_onnx_model:
            if self.use_compiled_fp16:
                wave = wave.half()
            models_logits = nn_models.run(
                None,
                {"input": wave.numpy()},
            )[0]
        elif is_openvino_model:
            if isinstance(nn_models, list):
                models_logits = []
                for nn_model in nn_models:
                    models_logits.append(nn_model([wave.numpy()])[nn_model.output(0)])
                models_logits = np.stack(models_logits, axis=0)
                models_logits = self._avarage_preds(models_logits)
            else:
                models_logits = nn_models([wave.numpy()])[nn_models.output(0)]
            if self.use_compiled_fp16:
                models_logits = models_logits.astype(np.float16)
        elif is_google_model:
            if self.device == "cuda":
                with tf.device("/gpu:0"):
                    google_model_output = nn_models.infer_tf(wave.numpy())
                    models_logits = google_postprocess(google_model_output)
            else:
                google_model_output = nn_models.infer_tf(wave.numpy())
                models_logits = google_postprocess(google_model_output)
        else:
            if self.model_output_key is None:
                models_logits = torch.stack(
                    [self._apply_pred_act(nn_model(wave).detach()).cpu() for nn_model in nn_models],
                    axis=0,
                )
            else:
                models_logits = torch.stack(
                    [
                        self._apply_pred_act(nn_model(wave)[self.model_output_key].detach().cpu())
                        for nn_model in nn_models
                    ],
                    axis=0,
                )
            models_logits = self._avarage_preds(models_logits)

        return models_logits

    @torch.no_grad()
    def predict_test_loader(
        self,
        nn_models,
        data_loader,
        is_onnx_model=False,
        is_openvino_model=False,
        is_google_model=False,
        google_postprocess=None,
    ):
        if not (is_onnx_model or is_openvino_model or is_google_model):
            for nn_model in nn_models:
                assert not nn_model.training

        test_model_logits = []
        test_dfidx = []
        test_end = []

        for wave, target, dfidx, start, end in self._tqdm_v(data_loader):
            models_logits = self._model_forward_test(
                nn_models=nn_models,
                wave=wave,
                is_onnx_model=is_onnx_model,
                is_openvino_model=is_openvino_model,
                is_google_model=is_google_model,
                google_postprocess=google_postprocess,
            )

            test_model_logits.append(models_logits)
            test_dfidx.append(dfidx.numpy())
            test_end.append(end.numpy())

        test_model_logits = np.concatenate(test_model_logits)
        test_dfidx = np.concatenate(test_dfidx)
        test_end = np.concatenate(test_end)

        assert len(test_model_logits.shape) == 2

        return test_model_logits, test_dfidx, test_end
