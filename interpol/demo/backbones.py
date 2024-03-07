from torch import nn
from functools import partial
from . import layers


class UNet(nn.Module):
    """A 2D UNet

    ```
    C -[conv xN]-> F ----------------------(cat)----------------------> 2*F -[conv xN]-> Cout
                   |                                                     ^
                   v                                                     |
                  F*m -[conv xN]-> F*m  ---(cat)---> 2*F*m -[conv xN]-> F*m
                                    |                  ^
                                    v                  |
                                  F*m*m -[conv xN]-> F*m*m
    ```
    """  # noqa: E501

    def __init__(
            self,
            ndim: int,
            nb_features: int = 16,
            mul_features: int = 2,
            nb_levels: int = 3,
            nb_conv_per_level: int = 2,
            activation: str = 'ReLU',
            pool: str = 'interpolate',
    ):
        """
        Parameters
        ----------
        ndim : int
            Number of spatial dimensions
        inp_channels : int
            Number of input channels
        out_channels : int
            Number of output chanels
        nb_features : int
            Number of features at the finest level
        mul_features : int
            Multiply the number of features by this number
            each time we go down one level.
        nb_conv_per_level : int
            Number of convolutional layers at each level.
        pool : {'interpolate', 'conv'}
            Method used to go down/up one level.
            If `interpolate`, use `torch.nn.functional.interpolate`.
            If `conv`, use strided convolutions on the way down, and
            transposed convolutions on the way up.
        activation : {'ReLU', 'ELU'}
            Type of activation
        """
        if pool == 'conv':
            DownKlass = layers.ConvDown
            UpKlass = layers.ConvUp
        else:
            DownKlass = layers.ConvDownResize
            UpKlass = layers.ConvUpResize

        make_inp = partial(
            layers.ConvBlockIn,
            ndim,
            activation=activation,
            nb_conv=nb_conv_per_level,
        )
        make_down = partial(
            DownKlass,
            ndim,
            activation=activation,
            nb_conv=nb_conv_per_level,
        )
        make_up = partial(
            UpKlass,
            ndim,
            activation=activation,
            nb_conv=nb_conv_per_level,
        )

        # number of features per level
        if isinstance(nb_features, int):
            F = [
                nb_features * mul_features**level for level in range(nb_levels)
            ]
        else:
            F = list(F)
            F += F[-1:] * max(0, nb_levels - len(F))

        downpath = [make_inp(F[0], F[0])]
        for i in range(1, nb_levels):
            downpath += [make_down(F[i-1], F[i])]
        F.reverse()
        uppath = []
        for i in range(nb_levels-1):
            uppath += [make_up(F[i], F[i+1])]

        super().__init__()
        self.downpath = nn.Sequential(*downpath)
        self.uppath = nn.Sequential(*uppath)
        print(self)

    def forward(self, inp, return_pyramid=False):
        """
        Parameters
        ----------
        inp : (B, in_channels, X, Y)
            Input tensor

        Returns
        -------
        out : (B, out_channels, X, Y)
            Output tensor
        """
        x, skips, pyramid = inp, [], []
        for layer in self.downpath:
            x = layer(x)
            skips.append(x)
        skips.pop(-1)  # no need to keep the corsest features
        if return_pyramid:
            pyramid += [x]
        for layer in self.uppath:
            x = layer(x, skips.pop(-1))
            if return_pyramid:
                pyramid += [x]
        return tuple(pyramid) if return_pyramid else x
