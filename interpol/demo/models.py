import torch
from torch import nn
from .backbones import UNet
from ..layers import SplineUp2, ResizeFlow


class VoxelMorph(nn.Sequential):
    """
    Construct a voxelmorph network with the given backbone
    """

    def __init__(self, ndim: int, **unet_parameters):
        """
        Parameters
        ----------
        ndim : int
            Number of spatial dimensions
        backbone_parameters : dict
            Parameters of the `UNet` class
        """
        Conv = getattr(nn, f'Conv{ndim}d')
        nf = unet_parameters.get('nb_features', 16)
        unet_parameters.setdefault('nb_levels', 5)
        unet_parameters.setdefault('nb_features', [16, 16, 32, 32, 32])
        unet_parameters.setdefault('activation', 'LeakyReLU')
        super().__init__(
            Conv(2, nf, kernel_size=[3]*ndim, padding='same'),
            UNet(ndim, **unet_parameters),
            Conv(nf, ndim, kernel_size=[1]*ndim),
        )

    def forward(self, fix, mov):
        """
        Predict a displacement field from a fixed and moving images

        Parameters
        ----------
        fix : (B, 1, X, Y) tensor
            Input fixed image
        mov : (B, 1, X, Y) tensor
            Input moving images

        Returns
        -------
        flow : (B, D, X, Y) tensor
            Predicted displacement field
        """
        fixmov = torch.cat([fix, mov], dim=1)
        return super().forward(fixmov)


class PyramidMorph(nn.Module):
    """
    Construct a network that predicts the flow in a pyramid fashion
    (inspired by LapIRN, but with a slightly different architecture)
    """

    def __init__(self, ndim: int, order: int = 3, **unet_parameters):
        """
        Parameters
        ----------
        ndim : int
            Number of spatial dimensions
        order : int
            Spline order encoding the flow
        backbone_parameters : dict
            Parameters of the `UNet` class
        """
        super().__init__()
        Conv = getattr(nn, f'Conv{ndim}d')
        unet_parameters.setdefault('nb_levels', 5)
        unet_parameters.setdefault('nb_features', [16, 16, 32, 32, 32])
        unet_parameters.setdefault('activation', 'LeakyReLU')
        nf = unet_parameters.get('nb_features', 16)
        mf = unet_parameters.get('mul_features', 2)
        np = unet_parameters.get('nb_levels', 5)
        if isinstance(nf, int):
            nf = [nf * mf**level for level in range(np)]
        else:
            nf = list(nf)
            nf += nf[-1:] * max(0, np - len(nf))
            nf = nf[:np]
        self.features = Conv(2, nf[0], kernel_size=[3]*ndim, padding='same')
        self.unet = UNet(ndim, **unet_parameters)
        nf.reverse()
        self.toflow = nn.ModuleList([
            Conv(nf[i], ndim, kernel_size=[1]*ndim) for i in range(np)
        ])
        self.up2 = (
            ResizeFlow(2, anchor='edges', bound='dft', extrapolate=True)
            if order == 1 else
            SplineUp2(interpolation=order)
        )

    def forward(self, fix, mov, return_pyramid=False):
        """
        Predict a displacement field from a fixed and moving images

        Parameters
        ----------
        fix : (B, 1, X, Y) tensor
            Input fixed image
        mov : (B, 1, X, Y) tensor
            Input moving images

        Returns
        -------
        flows : [list of] (B, D, X, Y) tensor
            Predicted spline coefficients of the flow fields
            (at each pyramid level, from coarse to fine)
        """
        fixmov = torch.cat([fix, mov], dim=1)
        # uncombined pyramid of features
        inppyr = self.unet(self.features(fixmov), return_pyramid=True)
        inppyr = list(inppyr)
        # feat2flow layers
        toflow = list(self.toflow)
        # coarsest flow
        flow = toflow.pop(0)(inppyr.pop(0))
        outpyr = [flow] if return_pyramid else []
        while inppyr:
            # upsample and add
            flow = self.up2(flow)
            flow += toflow.pop(0)(inppyr.pop(0))
            if return_pyramid:
                outpyr += [flow]
        return tuple(outpyr) if return_pyramid else flow
