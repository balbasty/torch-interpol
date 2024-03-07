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
        unet_parameters.setdefault('activation', 'LeakyReLU')
        super().__init__(
            Conv(2, nf, kernel_size=[3]*ndim, padding='same'),
            UNet(ndim, **unet_parameters),
            Conv(nf, ndim, kernel_size=[1]*ndim),
        )

    def forward(self, fixmov):
        """
        Predict a displacement field from a fixed and moving images

        Parameters
        ----------
        fixmov : (B, 2, X, Y) tensor
            Input fixed and moving images, stacked along
            the channel dimension

        Returns
        -------
        flow : (B, D, X, Y) tensor
            Predicted displacement field
        """
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
        nf = unet_parameters.get('nb_features', 16)
        np = unet_parameters.get('nb_levels', 3)
        unet_parameters.setdefault('nb_levels', 5)
        self.features = Conv(2, nf, kernel_size=[3]*ndim, padding='same')
        self.unet = UNet(ndim, **unet_parameters)
        self.toflow = nn.ModuleList([
            Conv(nf, ndim, kernel_size=[1]*ndim) for _ in range(np)
        ])
        self.up2 = (
            ResizeFlow(2, anchor='edges', bound='dft', extrapolate=True)
            if order == 1 else
            SplineUp2(interpolation=order)
        )

    def forward(self, fixmov):
        """
        Predict a displacement field from a fixed and moving images

        Parameters
        ----------
        fixmov : (B, 2, X, Y) tensor
            Input fixed and moving images, stacked along
            the channel dimension

        Returns
        -------
        flows : list[(B, D, Xl, Yl) tensor]
            Predicted spline coefficients of the flow fields, at each
            pyramid level, from coarse to fine.
        """
        # uncombined pyramid of flows
        inppyr = self.unet(self.features(fixmov), return_pyramid=True)
        flow = inppyr.pop(0)
        outpyr = [flow]
        while inppyr:
            flow = self.up2(flow)
            flow += inppyr.pop(0)
            outpyr += [flow]
        return outpyr
