import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional, Union


class ConvBlockIn(nn.Sequential):
    r"""A convolutional block at a single level

    ```
    Finp -[conv]-> Fout -[conv]-> ........ -[conv]-> Fout
                         \_______________________/
                                nb_conv - 1
    ```
    """

    def __init__(
            self,
            ndim: int,
            inp_channels: int,
            out_channels: Optional[int] = None,
            activation: str = 'ReLU',
            nb_conv: int = 1
    ):
        """
        Parameters
        ----------
        ndim : int
            Number of spatial dimensions
        inp_channels : int
            Number of input channels
        out_channels : int, default=inp_channels
            Number of output channels
        activation : str, default='ReLU'
            Activation function
        first_activation : bool
            Include very first activation
        final_activation: bool
            Include very last activation
        nb_conv : int, default=1
            Number of stacked convolutions
        """
        out_channels = out_channels or inp_channels
        Activation = getattr(nn, activation)
        Conv = getattr(nn, f'Conv{ndim}d')
        layers = [
            Conv(
                inp_channels,
                out_channels,
                kernel_size=[3]*ndim,
                padding='same',
            ),
            Activation()
        ]
        for _ in range(nb_conv-1):
            layers += [
                Conv(
                    out_channels,
                    out_channels,
                    kernel_size=[3]*ndim,
                    padding='same',
                ),
                Activation()
            ]
        super().__init__(*layers)


class ConvBlockOut(nn.Sequential):
    r"""A convolutional block at a single level

    ```
    Finp -[conv]-> ........ -[conv]-> Finp -[conv]-> Fout
          \_______________________/
                 nb_conv - 1
    ```
    """

    def __init__(
            self,
            ndim: int,
            inp_channels: int,
            out_channels: Optional[int] = None,
            activation: str = 'ReLU',
            final_activation: bool = False,
            nb_conv: int = 1,
            skip_channels: Union[int, bool] = True,
    ):
        """
        Parameters
        ----------
        ndim : int
            Number of spatial dimensions
        inp_channels : int
            Number of input channels
        out_channels : int, default=inp_channels
            Number of output channels
        activation : str, default='ReLU'
            Activation function
        first_activation : bool
            Include very first activation
        final_activation: bool
            Include very last activation
        nb_conv : int, default=1
            Number of stacked convolutions
        skip_channels : bool or int
            Number of additional channels from the skip connection.
            If `True`, same as `inp_channels`.
        """
        out_channels = out_channels or inp_channels
        if skip_channels is True:
            skip_channels = inp_channels
        elif skip_channels is False:
            skip_channels = 0
        Activation = getattr(nn, activation)
        Conv = getattr(nn, f'Conv{ndim}d')
        layers = []
        for nconv in range(nb_conv):
            true_inp_channels = inp_channels
            true_out_channels = inp_channels
            if nconv == 0:
                true_inp_channels += skip_channels
            if nconv == nb_conv-1:
                true_out_channels = out_channels
            layers += [
                Conv(
                    true_inp_channels,
                    true_out_channels,
                    kernel_size=[3]*ndim,
                    padding='same',
                ),
                Activation()
            ]
        if not final_activation:
            layers.pop(-1)
        super().__init__(*layers)


class ConvDown(nn.Sequential):
    r"""A convolutional block that starts with a strided conv

    ```
    Finp -[sconv]-> Fout -[conv]-> ........ -[conv]-> Fout
                          \_______________________/
                                    nb_conv
    ```
    """

    def __init__(
            self,
            ndim: int,
            inp_channels: int,
            out_channels: Optional[int] = None,
            activation: str = 'ReLU',
            nb_conv: int = 1,
    ):
        """
        Parameters
        ----------
        ndim : int
            Number of spatial dimensions
        inp_channels : int
            Number of input channels
        out_channels : int, default=inp_channels
            Number of output channels
        activation : str, default='ReLU'
            Activation function
        nb_conv : int, default=1
            Number of stacked convolutions (excluding the strided one)
        """
        out_channels = out_channels or inp_channels
        Activation = getattr(nn, activation)
        Conv = getattr(nn, f'Conv{ndim}d')
        layers = [
            Conv(
                inp_channels,
                out_channels,
                kernel_size=[2]*ndim,
                stride=[2]*ndim),
            Activation()
        ]
        for _ in range(nb_conv):
            layers += [
                Conv(
                    out_channels,
                    out_channels,
                    kernel_size=[3]*ndim,
                    padding='same',
                ),
                Activation()
            ]
        super().__init__(*layers)


class ConvUp(nn.Sequential):
    r"""A convolutional block that ends with a transposed conv

    ```
                     Fout
                       V
    Finp -[convt] -> [cat] -> 2*Fout -[conv]-> ........ -[conv]-> Fout
                                      \_______________________/
                                                nb_conv
    ```
    """

    def __init__(
            self,
            ndim: int,
            inp_channels: int,
            out_channels: Optional[int] = None,
            activation: str = 'ReLU',
            nb_conv: int = 1,
            skip_channels: Union[bool, int] = True,
    ):
        """
        Parameters
        ----------
        ndim : int
            Number of spatial dimensions
        inp_channels : int
            Number of input channels
        out_channels : int, default=inp_channels
            Number of output channels
        activation : str, default='ReLU'
            Activation function
        nb_conv : int, default=1
            Number of stacked convolutions (excluding the transposed one)
        skip_channels : bool or int
            Number of additional channels from the skip connection.
            If `True`, same as `out_channels`.
        """
        out_channels = out_channels or inp_channels
        if skip_channels is True:
            skip_channels = out_channels
        elif skip_channels is False:
            skip_channels = 0
        Activation = getattr(nn, activation)
        Conv = getattr(nn, f'Conv{ndim}d')
        ConvTranspose = getattr(nn, f'ConvTranspose{ndim}d')
        layers = [
            ConvTranspose(
                inp_channels,
                out_channels,
                kernel_size=[2]*ndim,
                stride=[2]*ndim,
            ),
            Activation()
        ]
        for nconv in range(nb_conv):
            true_inp_channels = out_channels
            if nconv == 0:
                true_inp_channels += skip_channels
            layers += [
                Conv(
                    true_inp_channels,
                    out_channels,
                    kernel_size=[3]*ndim,
                    padding='same',
                ),
                Activation()
            ]
        super().__init__(*layers)

    def forward(self, x, *skip):
        up, *convs = self
        x = up(x)
        if skip:
            x = torch.cat([x, *skip], dim=1)
        for conv in convs:
            x = conv(x)
        return x


class ConvBottleneck(nn.Sequential):
    r"""A convolutional bottleneck that starts with a strided conv and
    ends with a transposed conv

    ```
    Finp -[strided_conv]-> Fmid -[conv]-> ........ -[conv]-> Fmid -[transposed_conv]-> Finp
                                 \_______________________/
                                           nb_conv
    ```
    """  # noqa: E501

    def __init__(
            self,
            ndim: int,
            inp_channels: int,
            mid_channels: Optional[int] = None,
            activation: str = 'ReLU',
            nb_conv: int = 1,
    ):
        """
        Parameters
        ----------
        ndim : int
            Number of spatial dimensions
        inp_channels : int
            Number of input and output channels
        mid_channels : int, default=inp_channels
            Number of channels in the bottleneck
        activation : str, default='ReLU'
            Activation function
        nb_conv : int, default=1
            Number of stacked convolutions (excluding the strided ones)
        """
        mid_channels = mid_channels or inp_channels
        Activation = getattr(nn, activation)
        Conv = getattr(nn, f'Conv{ndim}d')
        ConvTranspose = getattr(nn, f'ConvTranspose{ndim}d')
        layers = [
            Conv(
                inp_channels,
                mid_channels,
                kernel_size=[2]*ndim,
                stride=[2]*ndim,
            ),
            Activation()
        ]
        for _ in range(nb_conv):
            layers += [
                Conv(
                    mid_channels,
                    mid_channels,
                    kernel_size=[3]*ndim,
                    padding='same',
                ),
                Activation()
            ]
        layers += [
            ConvTranspose(
                mid_channels,
                inp_channels,
                kernel_size=[2]*ndim,
                stride=[2]*ndim,
            ),
            Activation(),
        ]
        super().__init__(*layers)


class Resize(nn.Module):
    """Interpolation layer"""

    def __init__(self, factor: float = 2):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.factor, mode='bilinear')

    def __repr__(self):
        return type(self).__name__ + f'({self.factor})'


class ConvDownResize(nn.Sequential):
    r"""A convolutional block that starts with a downsampling layer

    ```
    Finp -[resize ÷2]-> Finp -[conv1]-> Fout -[conv]-> ........ -[conv]-> Fout
                                              \_______________________/
                                                        nb_conv
    ```
    """

    def __init__(
            self,
            ndim: int,
            inp_channels: int,
            out_channels: Optional[int] = None,
            activation: str = 'ReLU',
            nb_conv: int = 1,
    ):
        """
        Parameters
        ----------
        ndim : int
            Number of spatial dimensions
        inp_channels : int
            Number of input channels
        out_channels : int, default=inp_channels
            Number of output channels
        activation : str, default='ReLU'
            Activation function
        nb_conv : int, default=1
            Number of stacked convolutions
        """
        # NOTE: I merge the conv1 and ther first conv3+act into a single
        #       conv3+act (which has the same expressive power).
        out_channels = out_channels or inp_channels
        Activation = getattr(nn, activation)
        Conv = getattr(nn, f'Conv{ndim}d')
        layers = [
            Resize(0.5),
            Conv(
                inp_channels,
                out_channels,
                kernel_size=[3]*ndim,
                padding='same',
            ),
            Activation()
        ]
        for _ in range(nb_conv-1):
            layers += [
                Conv(
                    out_channels,
                    out_channels,
                    kernel_size=[3]*ndim,
                    padding='same',
                ),
                Activation()
            ]
        super().__init__(*layers)


class ConvUpResize(nn.Sequential):
    r"""A convolutional block that ends with an upsampling layer

    ```
                                         Fout
                                           v
     Finp -[conv1]-> Fout -[resize x2]-> [cat]-> 2*Fout -[conv]-> ........ -[conv]-> Fout
                                                         \_______________________/
                                                                   nb_conv
    ```
    """  # noqa: E501

    def __init__(
            self,
            ndim: int,
            inp_channels: int,
            out_channels: Optional[int] = None,
            activation: str = 'ReLU',
            nb_conv: int = 1,
            skip_channels: Union[int, bool] = True,
    ):
        """
        Parameters
        ----------
        ndim : int
            Number of spatial dimensions
        inp_channels : int
            Number of input channels (excluding skip connection)
        out_channels : int, default=inp_channels
            Number of output channels
        activation : str, default='ReLU'
            Activation function
        nb_conv : int, default=1
            Number of stacked convolutions (excluding the transposed one)
        skip_channels : bool or int
            Number of additional channels from the skip connection.
            If `True`, same as `inp_channels`.
        """
        out_channels = out_channels or inp_channels
        if skip_channels is True:
            skip_channels = out_channels
        elif skip_channels is False:
            skip_channels = 0
        Activation = getattr(nn, activation)
        Conv = getattr(nn, f'Conv{ndim}d')
        layers = [
            Conv(
                inp_channels,
                out_channels,
                kernel_size=[1]*ndim,
            ),
            Resize(2)
        ]
        for nconv in range(nb_conv-1):
            true_inp_channels = out_channels
            if nconv == 0:
                true_inp_channels += skip_channels
            layers += [
                Conv(
                    true_inp_channels,
                    out_channels,
                    kernel_size=[3]*ndim,
                    padding='same',
                ),
                Activation()
            ]
        super().__init__(*layers)

    def forward(self, x, *skip):
        conv1, resize, *layers = self
        x = resize(conv1(x))
        if skip:
            x = torch.cat([x, *skip], dim=1)
        for layer in layers:
            x = layer(x)
        return x


class ConvBottleneckResize(nn.Sequential):
    r"""A convolutional bottleneck that starts with a downsampling layer
    and ends with an upsampling layer

    ```
    Finp -[resize ÷2]-> Finp -[conv]-> Fmid ........ Fmid -[conv] -> Finp -[resize x2]-> Finp
                              \_________________________________/
                                            nb_conv
    ```
    """  # noqa: E501

    def __init__(
            self,
            ndim: int,
            inp_channels: int,
            mid_channels: Optional[int] = None,
            activation: str = 'ReLU',
            nb_conv: int = 1,
    ):
        """
        Parameters
        ----------
        ndim : int
            Number of spatial dimensions
        inp_channels : int
            Number of input and output channels
        mid_channels : int, default=inp_channels
            Number of channels in the bottleneck
        activation : str, default='ReLU'
            Activation function
        nb_conv : int, default=1
            Number of stacked convolutions (excluding the strided ones)
        """
        mid_channels = mid_channels or inp_channels
        Activation = getattr(nn, activation)
        Conv = getattr(nn, f'Conv{ndim}d')
        if nb_conv < 2:
            mid_channels = inp_channels
        layers = [
            Resize(0.5),
            Conv(
                inp_channels,
                mid_channels,
                kernel_size=[3]*ndim,
                padding='same',
            ),
            Activation()
        ]
        for _ in range(nb_conv-2):
            layers += [
                Conv(
                    mid_channels,
                    mid_channels,
                    kernel_size=[3]*ndim,
                    padding='same',
                ),
                Activation()
            ]
        if nb_conv > 1:
            layers += [
                Conv(
                    mid_channels,
                    inp_channels,
                    kernel_size=[3]*ndim,
                    padding='same',
                ),
                Activation(),
            ]
        layers += [Resize(2)]
        super().__init__(*layers)
