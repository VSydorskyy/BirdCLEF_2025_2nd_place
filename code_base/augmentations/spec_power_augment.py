import numpy as np
import torch.nn as nn

from ..models.blocks import NormalizeMelSpec


class RandomSpecPower(nn.Module):
    def __init__(self, power_range, p=1.0, inplace=True, exportable=True):
        super().__init__()
        self.power_range = power_range
        self.p = p
        self.inplace = inplace

        self.norm = NormalizeMelSpec(exportable=exportable)

    def forward(self, x):
        # B, C, H, W
        assert len(x.size()) == 4
        assert x.size(1) == 1
        if not self.inplace:
            output = x.clone()
        for i in range(x.shape[0]):
            if np.random.binomial(n=1, p=self.p):
                power = np.random.uniform(*self.power_range)
                if self.inplace:
                    x[i : i + 1] = x[i : i + 1] ** power
                else:
                    output[i : i + 1] = output[i : i + 1] ** power
        if self.inplace:
            x[:, 0, :, :] = self.norm(x[:, 0, :, :])
            return x
        else:
            output[:, 0, :, :] = self.norm(output[:, 0, :, :])
            return output
