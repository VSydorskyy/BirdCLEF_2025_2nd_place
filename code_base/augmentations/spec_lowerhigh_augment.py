import random

import numpy as np
import torch
import torch.nn as nn

from ..models.blocks import NormalizeMelSpec


class RandomLowerHighFreq(nn.Module):
    def __init__(self, p=1.0, inplace=True, exportable=True):
        super().__init__()
        self.p = p
        self.inplace = inplace

        self.norm = NormalizeMelSpec(exportable=exportable)

    @staticmethod
    def _create_pink_noise(n_mels):
        r = random.randint(n_mels // 2, n_mels)
        x = random.random() / 2
        pink_noise = np.array([np.concatenate((1 - np.arange(r) * x / r, np.zeros(n_mels - r) - x + 1))]).T
        return torch.from_numpy(pink_noise)

    def forward(self, x):
        # B, C, Freq, Time
        assert len(x.size()) == 4
        assert x.size(1) == 1
        if not self.inplace:
            output = x.clone()
        for i in range(x.shape[0]):
            if np.random.binomial(n=1, p=self.p):
                pink_noise = self._create_pink_noise(x.size(2)).to(x)
                if self.inplace:
                    x[i : i + 1] = x[i : i + 1] * pink_noise
                else:
                    output[i : i + 1] = output[i : i + 1] * pink_noise
        if self.inplace:
            x[:, 0, :, :] = self.norm(x[:, 0, :, :])
            return x
        else:
            output[:, 0, :, :] = self.norm(output[:, 0, :, :])
            return output
