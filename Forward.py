import torch.nn as nn


class Feed_Forward(nn.Module):
    def __init__(self, d_middle, d_FF_out):
        super().__init__()

        self.d_middle = d_middle
        self.d_FF_out = d_FF_out

    def forward(self, x):
        A = nn.Linear(x.shape[-1], self.d_middle).cuda()(x)
        B = nn.ReLU()(A)
        C = nn.Linear(self.d_middle, self.d_FF_out).cuda()(B)
        return C
