from torch import nn
from . import api, energies
from .splines import to_int as inter_to_int
from .resize import resize
from .restrict import restrict
from .jit_utils import movedim1
from .utils import make_list


class GridPull(nn.Module):
    """
    Deform an image using a coordinates field.
    """

    def __init__(
            self,
            interpolation='linear',
            bound='zero',
            extrapolate=False,
            prefilter=False
    ):
        """

        Notes
        -----
        {interpolation}

        {bound}

        Parameters
        ----------
        interpolation : int or sequence[int]
            Interpolation order.
        bound : BoundType or sequence[BoundType]
            Boundary conditions.
        extrapolate : bool or int
            Extrapolate out-of-bound data.
        prefilter : bool
            Apply spline pre-filter (= interpolates the input)

        """
        super().__init__()
        self.interpolation = interpolation
        self.bound = bound
        self.extrapolate = extrapolate
        self.prefilter = prefilter

    @property
    def _options(self):
        return dict(
            interpolation=self.interpolation,
            bound=self.bound,
            extrapolate=self.extrapolate,
            prefilter=self.prefilter,
        )

    def forward(self, input, grid):
        """
        Sample an image.

        If the input dtype is not a floating point type, the input image is
        assumed to contain labels. Then, unique labels are extracted
        and resampled individually, making them soft labels. Finally,
        the label map is reconstructed from the individual soft labels by
        assigning the label with maximum soft value.

        Parameters
        ----------
        input : (batch, channel, *inshape) tensor
            Input image.
        grid : (batch, ndim, *outshape) tensor
            Coordinate field, in voxels.

        Returns
        -------
        output : (batch, channel, *outshape) tensor
            Deformed image.
        """
        grid = movedim1(grid, 1, -1)
        return api.grid_pull(input, grid, **self._options)


class FlowPull(GridPull):
    """
    Deform an image using a displacement field
    """

    def forward(self, input, flow):
        """
        Sample an image.

        If the input dtype is not a floating point type, the input image is
        assumed to contain labels. Then, unique labels are extracted
        and resampled individually, making them soft labels. Finally,
        the label map is reconstructed from the individual soft labels by
        assigning the label with maximum soft value.

        Parameters
        ----------
        input : (batch, channel, *inshape) tensor
            Input image.
        flow : (batch, ndim, *outshape) tensor
            Displacement field, in voxels.

        Returns
        -------
        output : (batch, channel, *outshape) tensor
            Deformed image.
        """
        flow = movedim1(flow, 1, -1)
        flow = api.add_identity_grid(flow)
        flow = movedim1(flow, -1, 1)
        return super().forward(input, flow)


class GridPush(nn.Module):
    """
    Splat an image using a coordinates field.
    """

    def __init__(
            self,
            interpolation='linear',
            bound='zero',
            extrapolate=False,
            prefilter=False
    ):
        """

        Notes
        -----
        {interpolation}

        {bound}

        Parameters
        ----------
        interpolation : int or sequence[int]
            Interpolation order.
        bound : BoundType or sequence[BoundType]
            Boundary conditions.
        extrapolate : bool or int
            Extrapolate out-of-bound data.
        prefilter : bool
            Apply spline pre-filter (= interpolates the input)

        """
        super().__init__()
        self.interpolation = interpolation
        self.bound = bound
        self.extrapolate = extrapolate
        self.prefilter = prefilter

    @property
    def _options(self):
        return dict(
            interpolation=self.interpolation,
            bound=self.bound,
            extrapolate=self.extrapolate,
            prefilter=self.prefilter,
        )

    def forward(self, input, grid, shape=None):
        """
        Splat an image.

        Parameters
        ----------
        input : (batch, channel, *inshape) tensor
            Input image.
        grid : (batch, ndim, *inshape) tensor
            Coordinate field, in voxels.
        shape : sequence[int], default=inshape
            Output spatial shape.

        Returns
        -------
        output : (batch, channel, *shape) tensor
            Splatted image.
        """
        grid = movedim1(grid, 1, -1)
        return api.grid_push(input, grid, shape, **self._options)


class FlowPush(GridPush):
    """
    Splat an image using a displacement field
    """

    def forward(self, input, flow, shape=None):
        """
        Splat an image.

        Parameters
        ----------
        input : (batch, channel, *inshape) tensor
            Input image.
        flow : (batch, ndim, *inshape) tensor
            Displacement field, in voxels.
        shape : sequence[int], default=inshape
            Output spatial shape

        Returns
        -------
        output : (batch, channel, *shape) tensor
            Deformed image.
        """
        flow = movedim1(flow, 1, -1)
        flow = api.add_identity_grid(flow)
        flow = movedim1(flow, -1, 1)
        return super().forward(input, flow, shape)


class Resize(nn.Module):
    """
    Resize (interpolate) an image
    """

    def __init__(
            self,
            factor=None,
            shape=None,
            anchor='edge',
            interpolation='linear',
            bound='zero',
            extrapolate=False,
            prefilter=True
    ):
        """
        Notes
        -----
        {interpolation}

        {bound}

        * A least one of `factor` and `shape` must be specified
        * If `anchor in ('center', 'edge')`, exactly one of `factor`
          or `shape` must be specified.
        * If `anchor in ('first', 'last')`, `factor` must be provided
          even if `shape` is specified.
        *  Because of rounding, it is in general not assured that
          `resize(resize(x, f), 1/f)` returns a tensor with the same
          shape as x.

        ```
             edge           center          first           last
        e - + - + - e   + - + - + - +   + - + - + - +   + - + - + - +
        | . | . | . |   | c | . | c |   | f | . | . |   | . | . | . |
        + _ + _ + _ +   + _ + _ + _ +   + _ + _ + _ +   + _ + _ + _ +
        | . | . | . |   | . | . | . |   | . | . | . |   | . | . | . |
        + _ + _ + _ +   + _ + _ + _ +   + _ + _ + _ +   + _ + _ + _ +
        | . | . | . |   | c | . | c |   | . | . | . |   | . | . | l |
        e _ + _ + _ e   + _ + _ + _ +   + _ + _ + _ +   + _ + _ + _ +
        ```

        Parameters
        ----------
        factor : float or list[float], optional
            Resizing factor

            * > 1 : larger image <-> smaller voxels
            * < 1 : smaller image <-> larger voxels
        shape : (ndim,) list[int], optional
            Output shape
        anchor : {'center', 'edge', 'first', 'last'} or list
            * In cases 'c' and 'e', the volume shape is multiplied by the
              zoom factor (and eventually truncated), and two anchor points
              are used to determine the voxel size.
            * In cases 'f' and 'l', a single anchor point is used so that
              the voxel size is exactly divided by the zoom factor.
              This case with an integer factor corresponds to subslicing
              the volume (e.g., `vol[::f, ::f, ::f]`).
            * A list of anchors (one per dimension) can also be provided.
        interpolation : int or sequence[int]
            Interpolation order.
        bound : BoundType or sequence[BoundType]
            Boundary conditions.
        extrapolate : bool or int
            Extrapolate out-of-bound data.
        prefilter : bool
            Apply spline pre-filter (= interpolates the input)

        """
        super().__init__()
        self.factor = factor
        self.shape = shape
        self.anchor = anchor
        self.interpolation = interpolation
        self.bound = bound
        self.extrapolate = extrapolate
        self.prefilter = prefilter

    @property
    def _options(self):
        return dict(
            factor=self.factor,
            shape=self.shape,
            anchor=self.anchor,
            interpolation=self.interpolation,
            bound=self.bound,
            extrapolate=self.extrapolate,
            prefilter=self.prefilter,
        )

    def forward(self, input, **kwargs):
        """
        Resize an image

        Parameters
        ----------
        input : (batch, channel, *inshape)
            Input image

        Other Parameters
        ----------------
        shape : (ndim,) list[int], optional
            Output shape. If not provided at call time, use self.shape

        Parameters
        ----------
        output : (batch, channel, *shape)
            Resized image
        """
        options = self._options
        options.update(kwargs)
        return resize(input, **options)


class Restrict(nn.Module):
    """
    Restrict an image (adjoint of resize)
    """

    def __init__(
            self,
            factor=None,
            shape=None,
            anchor='edge',
            interpolation='linear',
            bound='zero',
            reduce_sum=False,
    ):
        """
        Notes
        -----
        {interpolation}

        {bound}

        * A least one of `factor` and `shape` must be specified
        * If `anchor in ('center', 'edge')`, exactly one of `factor`
          or `shape` must be specified.
        * If `anchor in ('first', 'last')`, `factor` must be provided
          even if `shape` is specified.
        *  Because of rounding, it is in general not assured that
          `resize(resize(x, f), 1/f)` returns a tensor with the same
          shape as x.

        ```
            edge           center           first           last
        e - + - + - e   + - + - + - +   + - + - + - +   + - + - + - +
        | . | . | . |   | c | . | c |   | f | . | . |   | . | . | . |
        + _ + _ + _ +   + _ + _ + _ +   + _ + _ + _ +   + _ + _ + _ +
        | . | . | . |   | . | . | . |   | . | . | . |   | . | . | . |
        + _ + _ + _ +   + _ + _ + _ +   + _ + _ + _ +   + _ + _ + _ +
        | . | . | . |   | c | . | c |   | . | . | . |   | . | . | l |
        e _ + _ + _ e   + _ + _ + _ +   + _ + _ + _ +   + _ + _ + _ +
        ```

        Parameters
        ----------
        factor : float or list[float], optional
            Resizing factor

            * > 1 : larger image <-> smaller voxels
            * < 1 : smaller image <-> larger voxels
        shape : (ndim,) list[int], optional
            Output shape
        anchor : {'center', 'edge', 'first', 'last'} or list
            * In cases 'c' and 'e', the volume shape is multiplied by the
              zoom factor (and eventually truncated), and two anchor points
              are used to determine the voxel size.
            * In cases 'f' and 'l', a single anchor point is used so that
              the voxel size is exactly divided by the zoom factor.
              This case with an integer factor corresponds to subslicing
              the volume (e.g., `vol[::f, ::f, ::f]`).
            * A list of anchors (one per dimension) can also be provided.
        interpolation : int or sequence[int]
            Interpolation order.
        bound : BoundType or sequence[BoundType]
            Boundary conditions.
        reduce_sum : bool
            Do not normalize by the number of accumulated values per voxel

        """
        super().__init__()
        self.factor = factor
        self.shape = shape
        self.anchor = anchor
        self.interpolation = interpolation
        self.bound = bound
        self.reduce_sum = reduce_sum

    @property
    def _options(self):
        return dict(
            factor=self.factor,
            shape=self.shape,
            anchor=self.anchor,
            interpolation=self.interpolation,
            bound=self.bound,
            reduce_sum=self.reduce_sum,
        )

    def forward(self, input, **kwargs):
        """
        Restrict an image

        Parameters
        ----------
        input : (batch, channel, *inshape)
            Input image

        Other Parameters
        ----------------
        shape : (ndim,) list[int], optional
            Output shape. If not provided at call time, use self.shape

        Parameters
        ----------
        output : (batch, channel, *shape)
            Restricted image
        """
        options = self._options
        options.update(kwargs)
        return restrict(input, **options)


class ResizeFlow(Resize):
    """
    Resize (interpolate) a displacement field
    """

    def __init__(
            self,
            factor=None,
            shape=None,
            anchor='edge',
            interpolation='linear',
            bound='dft',
            extrapolate=False,
            prefilter=True,
    ):
        super().__init__(
            factor=factor,
            shape=shape,
            anchor=anchor,
            interpolation=interpolation,
            bound=bound,
            extrapolate=extrapolate,
            prefilter=prefilter,
        )

    def forward(self, flow, **kwargs):
        """
        Resize a displacement field. The magnitude of the displacements
        gets rescaled as well.

        Parameters
        ----------
        flow : (batch, ndim, *inshape)
            Input displacement field

        Other Parameters
        ----------------
        shape : (ndim,) list[int], optional
            Output shape. If not provided at call time, use self.shape

        Parameters
        ----------
        output : (batch, ndim, *shape)
            Resized displacement field
        """
        ishape = flow.shape[2:]
        flow = super().forward(flow)
        oshape = flow.shape[2:]
        anchor = self.anchor[0].lower()
        if anchor == 'c':
            for isz, osz, flow1 in zip(ishape, oshape, flow.unbind(1)):
                flow1 *= (osz - 1) / (isz - 1)
        elif anchor == 'e':
            for isz, osz, flow1 in zip(ishape, oshape, flow.unbind(1)):
                flow1 *= osz / isz
        else:
            factor = make_list(self.factor, len(ishape))
            for f, flow1 in zip(factor, flow.unbind(1)):
                flow1 *= f
        return flow


class ValueToCoeff(nn.Module):
    """Compute spline coefficients from values"""

    def __init__(
            self,
            interpolation='linear',
            bound='zero',
    ):
        """
        Notes
        -----
        {interpolation}

        {bound}

        Parameters
        ----------
        interpolation : int or sequence[int]
            Interpolation order.
        bound : BoundType or sequence[BoundType]
            Boundary conditions.
        """
        super().__init__()
        self.interpolation = interpolation if interpolation != 'fd' else 1
        self.bound = bound

    @property
    def _options(self):
        return dict(
            interpolation=self.interpolation,
            bound=self.bound,
        )

    def forward(self, input):
        """
        Parameters
        ----------
        input : (batch, channel, *shape) tensor
            Input image of values

        Returns
        -------
        output : (batch, channel, *shape) tensor
            Input image of spline coefficients
        """
        ndim = self.input.ndim - 2
        return api.spline_coeff_nd(input, **self._options, dim=ndim)


class CoeffToValue(nn.Module):
    """Compute values from spline coefficients"""

    def __init__(
            self,
            interpolation='linear',
            bound='zero',
    ):
        """
        Notes
        -----
        {interpolation}

        {bound}

        Parameters
        ----------
        interpolation : int or sequence[int]
            Interpolation order.
        bound : BoundType or sequence[BoundType]
            Boundary conditions.
        """
        super().__init__()
        self.interpolation = interpolation if interpolation != 'fd' else 1
        self.bound = bound

    @property
    def _options(self):
        return dict(
            interpolation=self.interpolation,
            bound=self.bound,
            extrapolate=True,
            prefilter=False,
        )

    def forward(self, input):
        """
        Parameters
        ----------
        input : (batch, channel, *shape) tensor
            Input image of spline coefficients

        Returns
        -------
        output : (batch, channel, *shape) tensor
            Input image of values
        """
        grid = api.identity_grid(
            input.shape[2:], dtype=input.dtype, device=input.device)
        return api.grid_pull(input, grid, **self._options)


class FlowExp(nn.Module):
    """Exponentiate a stationary velocity field"""

    def __init__(
            self,
            nsteps=8,
            interpolation='linear',
            bound='dft',
            extrapolate=False,
            coeff=False,
    ):
        """

        Notes
        -----
        {interpolation}

        {bound}

        Parameters
        ----------
        nsteps : int
            Number of scaling and squaring steps
        interpolation : int or sequence[int]
            Interpolation order.
        bound : BoundType or sequence[BoundType]
            Boundary conditions.
        extrapolate : bool or int
            Extrapolate out-of-bound data.
        coeff : bool
            If True, the input velocity image contains spline coefficients,
            and spline coefficients will also be returned.
            If False, the input velocity image contains actual values,
            and values will also be returned.

        """
        super().__init__()
        self.nsteps = nsteps
        self.interpolation = interpolation
        self.bound = bound
        self.extrapolate = extrapolate
        self.coeff = coeff

    @property
    def _options(self):
        return dict(
            interpolation=self.interpolation,
            bound=self.bound,
            extrapoalte=self.extrapolate,
        )

    def forward(self, flow):
        """
        Exponentiate the SVF

        Parameters
        ----------
        flow : (batch, ndim, *shape) tensor
            Stationary velocity field
        """
        # helpers
        to_coeff = ValueToCoeff(
            interpolation=self.interpolation, bound=self.bound)
        from_coeff = CoeffToValue(
            interpolation=self.interpolation, bound=self.bound)
        compose = FlowPull(
            interpolation=self.interpolation, bound=self.bound,
            extrapolate=self.extrapolate, prefilter=False)
        flow = flow / 2**self.nsteps
        # init
        if not self.coeff:
            coeff = to_coeff(flow)
        else:
            coeff = flow
            flow = from_coeff(coeff)
        # scale and square
        for _ in range(self.nsteps):
            flow = compose(coeff, flow)
            coeff = to_coeff(flow)
        # final
        return coeff if self.coeff else flow


class FlowMomentum(nn.Module):
    """Compute the momentum of a displacement field"""

    def __init__(
            self,
            absolute=0,
            membrane=0,
            bending=0,
            div=0,
            shears=0,
            norm=True,
            interpolation='linear',
            bound='dft',
    ):
        """
        Parameters
        ----------
        absolute : float
            Penalty on absolute displacement
        membrane : float
            Penalty on first derivatives
        bending : float
            Penalty on second derivatives
        div : float
            Penalty on volume changes
        shears : float
            Penalty on shears
        norm : bool
            If True, compute the average energy across the field of view.
            Otherwise, compute the sum (integral) of the energy across the FOV.
        interpolation : int
            Spline order
        bound : bound_like
            Boundary conditions
        """
        super().__init__()
        self.absolute = absolute
        self.membrane = membrane
        self.bending = bending
        self.div = div
        self.shears = shears
        self.norm = norm
        self.interpolation = interpolation
        self.bound = bound

    @property
    def _options(self):
        return dict(
            absolute=self.absolute,
            membrane=self.membrane,
            bending=self.bending,
            div=self.div,
            shears=self.shears,
            norm=self.norm,
            order=inter_to_int(self.interpolation)
            if self.interpolation != 'fd' else self.interpolation,
            bound=self.bound,
        )

    def forward(self, flow):
        """
        Parameters
        ----------
        flow : (batch, ndim, *shape) tensor
            Spline coefficients of a displacement field

        Returns
        -------
        mom : (batch, ndim, *shape) tensor
            Momentum field
        """
        flow = movedim1(flow, 1, -1)
        flow = energies.flowmom(flow, **self._options)
        flow = movedim1(flow, -1, 1)
        return flow


class FlowLoss(FlowMomentum):
    """Compute the regularization loss of a displacement field"""

    def forward(self, flow):
        """
        Parameters
        ----------
        flow : (batch, ndim, *shape) tensor
            Spline coefficients of a displacement field

        Returns
        -------
        loss : scalar tensor
            loss -- averaged across batch elements
        """
        mom = super().forward(flow)
        nbatch = len(flow)
        return mom.flatten().dot(flow.flatten()) / nbatch


class SplineUp2(nn.Module):
    """MSE-minmizing upsampling of a displacement field -- by a factor 2"""

    def __init__(
            self,
            interpolation='linear',
            bound='dft',
    ):
        """
        Parameters
        ----------
        interpolation : int
            Spline order
        bound : bound_like
            Boundary conditions
        """
        super().__init__()
        self.interpolation = interpolation
        self.bound = bound

    @property
    def _options(self):
        return dict(
            order=inter_to_int(self.interpolation),
            bound=self.bound,
        )

    def forward(self, flow):
        """
        Parameters
        ----------
        flow : (batch, ndim, *shape) tensor
            Spline coefficients of a displacement field

        Returns
        -------
        flow2 : (batch, ndim, *shape_twice) tensor
            Spline coefficients of a larger displacement field
        """
        flow = movedim1(flow, 1, -1)
        flow = energies.flow_upsample2(flow, **self._options)
        flow = movedim1(flow, -1, 1)
        return flow


GridPull.__init__.__doc__.format(
    bound=api._doc_bound,
    interpolation=api._doc_interpolation,
)
GridPush.__init__.__doc__.format(
    bound=api._doc_bound,
    interpolation=api._doc_interpolation,
)
