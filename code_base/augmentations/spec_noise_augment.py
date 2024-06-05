import random

import numpy as np
import torch
import torch.nn as nn

from ..models.blocks import NormalizeMelSpec


class RandomNoise(nn.Module):
    def __init__(self, p=1.0, noise_type="white", bandwidth=20, noise_scale=(0.05, 0.1), inplace=True, exportable=True):
        assert noise_type in ["white", "bandpass"]
        assert len(noise_scale) == 2
        assert noise_scale[0] < noise_scale[1]
        super().__init__()
        self.p = p
        self.inplace = inplace
        self.noise_scale = noise_scale
        self.noise_type = noise_type
        self.bandwidth = bandwidth

        self.norm = NormalizeMelSpec(exportable=exportable)

    def _apply_white_noise(self, input_spec):
        print("White")
        return input_spec + (torch.randn_like(input_spec) * random.uniform(*self.noise_scale))

    def _apply_bandpass_noise(self, input_spec):
        print("Bandpass")
        a = random.randint(0, input_spec.shape[2] // 2)
        b = random.randint(a + self.bandwidth, input_spec.shape[2])
        input_spec[:, :, a:b, :] = self._apply_white_noise(input_spec[:, :, a:b, :])
        return input_spec

    def forward(self, x):
        # B, C, Freq, Time
        assert len(x.size()) == 4
        assert x.size(1) == 1
        if not self.inplace:
            output = x.clone()
        for i in range(x.shape[0]):
            if np.random.binomial(n=1, p=self.p):
                if self.inplace:
                    if self.noise_type == "white":
                        x[i : i + 1] = self._apply_white_noise(x[i : i + 1])
                    elif self.noise_type == "bandpass":
                        x[i : i + 1] = self._apply_bandpass_noise(x[i : i + 1])
                else:
                    if self.noise_type == "white":
                        output[i : i + 1] = self._apply_white_noise(output[i : i + 1])
                    elif self.noise_type == "bandpass":
                        output[i : i + 1] = self._apply_bandpass_noise(output[i : i + 1])
        if self.inplace:
            x[:, 0, :, :] = self.norm(x[:, 0, :, :])
            return x
        else:
            output[:, 0, :, :] = self.norm(output[:, 0, :, :])
            return output
