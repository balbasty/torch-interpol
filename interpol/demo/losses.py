from torch import nn


class NCC(nn.Module):

    def forward(self, x, y):

        x = x.reshape(x.shape[:2] + (-1,))
        y = y.reshape(y.shape[:2] + (-1,))

        mux, muy = x.mean(-1, keepdim=True), y.mean(-1, keepdim=True)
        sdx, sdy = x.std(-1, keepdim=True), y.std(-1, keepdim=True)

        cc = ((x - mux) * (y - muy)).mean(-1, keepdim=True) / (sdx * sdy)
        return (1 - cc.square()).mean()
