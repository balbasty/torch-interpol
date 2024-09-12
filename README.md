# torch-interpol
High-order spline interpolation in PyTorch

## Description

This package contains a pure python implementation of **high-order spline
interpolation** for ND tensors (including 2D and 3D images). It makes use
of the just-in-time capabilities of TorchScript and explicitly implements
the forward and backward passes of all functions, making it **fast** and
**memory-efficient**.

All the functions available in this (small) package were originally
implemented in [NITorch](https://github/balbasty/nitorch), a larger
PyTorch-based package dedicated to NeuroImaging and Medical Image Computing.

## Installation

### Dependency

- `torch >= 1.3`
- `torch >= 1.8` if any of these functions is used:
    - `flowiconv`
    - `flowimom`
    - `flow_upsample2`
    - `coeff_upsample2`
- `torch-bounds >= 0.1`

### Conda

```shell
conda install torch-interpol -c balbasty -c pytorch
```

### Pip

```shell
pip install torch-interpol
```

## Usage

**See our [example notebooks](examples/)**

## Quick doc

```
Notes
-----

`interpolation` can be an int, a string or an InterpolationType.
Possible values are:
    - 0 or 'nearest'
    - 1 or 'linear'
    - 2 or 'quadratic'
    - 3 or 'cubic'
    - 4 or 'fourth'
    - 5 or 'fifth'
    - etc.
A list of values can be provided, in the order [W, H, D],
to specify dimension-specific interpolation orders.

`bound` can be an int, a string or a BoundType.
Possible values are:
    - 'replicate'  or 'nearest'     :  a  a  a  |  a  b  c  d  |  d  d  d
    - 'dct1'       or 'mirror'      :  d  c  b  |  a  b  c  d  |  c  b  a
    - 'dct2'       or 'reflect'     :  c  b  a  |  a  b  c  d  |  d  c  b
    - 'dst1'       or 'antimirror'  : -b -a  0  |  a  b  c  d  |  0 -d -c
    - 'dst2'       or 'antireflect' : -c -b -a  |  a  b  c  d  | -d -c -b
    - 'dft'        or 'wrap'        :  b  c  d  |  a  b  c  d  |  a  b  c
    - 'zero'       or 'zeros'       :  0  0  0  |  a  b  c  d  |  0  0  0
A list of values can be provided, in the order [W, H, D],
to specify dimension-specific boundary conditions.

Note that
- `dft` corresponds to circular padding
- `dct2` corresponds to Neumann boundary conditions (symmetric)
- `dst2` corresponds to Dirichlet boundary conditions (antisymmetric)
See https://en.wikipedia.org/wiki/Discrete_cosine_transform
    https://en.wikipedia.org/wiki/Discrete_sine_transform
```

### Spline interpolation

```python
interpol.grid_pull(
    input,
    grid,
    interpolation='linear',
    bound='zero',
    extrapolate=False,
    prefilter=False,
)
"""
Sample an image with respect to a deformation field.

If the input dtype is not a floating point type, the input image is
assumed to contain labels. Then, unique labels are extracted
and resampled individually, making them soft labels. Finally,
the label map is reconstructed from the individual soft labels by
assigning the label with maximum soft value.

Parameters
----------
input : (..., [channel], *inshape) tensor
    Input image.
grid : (..., *outshape, dim) tensor
    Transformation field.
interpolation : int or sequence[int], default=1
    Interpolation order.
bound : BoundType or sequence[BoundType], default='zero'
    Boundary conditions.
extrapolate : bool or int, default=True
    Extrapolate out-of-bound data.
prefilter : bool, default=False
    Apply spline pre-filter (= interpolates the input)

Returns
-------
output : (..., [channel], *outshape) tensor
    Deformed image.
"""
```

```python
interpol.grid_push(
    input,
    grid,
    shape=None,
    interpolation='linear',
    bound='zero',
    extrapolate=False,
    prefilter=False,
)
"""
Splat an image with respect to a deformation field.

This function is the exact numerical adjoint of `grid_pull`.

Parameters
----------
input : (..., [channel], *inshape) tensor
    Input image.
grid : (..., *inshape, dim) tensor
    Transformation field.
shape : sequence[int], default=inshape
    Output shape
interpolation : int or sequence[int], default=1
    Interpolation order.
bound : BoundType, or sequence[BoundType], default='zero'
    Boundary conditions.
extrapolate : bool or int, default=True
    Extrapolate out-of-bound data.
prefilter : bool, default=False
    Apply spline pre-filter.

Returns
-------
output : (..., [channel], *shape) tensor
    Spatted image.
"""
```

```python
interpol.grid_grad(
    input,
    grid,
    interpolation='linear',
    bound='zero',
    extrapolate=False,
    prefilter=False,
)
"""
Sample spatial gradients of an image with respect to a deformation field.

Parameters
----------
input : (..., [channel], *inshape) tensor
    Input image.
grid : (..., *inshape, dim) tensor
    Transformation field.
shape : sequence[int], default=inshape
    Output shape
interpolation : int or sequence[int], default=1
    Interpolation order.
bound : BoundType, or sequence[BoundType], default='zero'
    Boundary conditions.
extrapolate : bool or int, default=True
    Extrapolate out-of-bound data.
prefilter : bool, default=False
    Apply spline pre-filter (= interpolates the input)

Returns
-------
output : (..., [channel], *shape, dim) tensor
    Sampled gradients.
"""
```

```python
interpol.spline_coeff_nd(
    input,
    interpolation='linear',
    bound='dct2',
    dim=None,
    inplace=False,
)
"""
Compute the interpolating spline coefficients, for a given spline order
and boundary conditions, along the last `dim` dimensions.

References
----------
..[1]  M. Unser, A. Aldroubi and M. Eden.
       "B-Spline Signal Processing: Part I-Theory,"
       IEEE Transactions on Signal Processing 41(2):821-832 (1993).
..[2]  M. Unser, A. Aldroubi and M. Eden.
       "B-Spline Signal Processing: Part II-Efficient Design and Applications,"
       IEEE Transactions on Signal Processing 41(2):834-848 (1993).
..[3]  M. Unser.
       "Splines: A Perfect Fit for Signal and Image Processing,"
       IEEE Signal Processing Magazine 16(6):22-38 (1999).

Parameters
----------
input : (..., *spatial) tensor
    Input image.
interpolation : int or sequence[int], default=1
    Interpolation order.
bound : BoundType or sequence[BoundType], default='dct1'
    Boundary conditions.
dim : int, default=-1
    Number of spatial dimensions
inplace : bool, default=False
    Process the volume in place.

Returns
-------
output : (..., *spatial) tensor
    Coefficient image.
"""
```

```python
interpol.resize(
    image,
    factor=None,
    shape=None,
    anchor='c',
    interpolation=1,
    prefilter=True
)
"""Resize an image by a factor or to a specific shape.

Notes
-----
.. A least one of `factor` and `shape` must be specified
.. If `anchor in ('centers', 'edges')`, exactly one of `factor` or
   `shape must be specified.
.. If `anchor in ('first', 'last')`, `factor` must be provided even
   if `shape` is specified.
.. Because of rounding, it is in general not assured that
   `resize(resize(x, f), 1/f)` returns a tensor with the same shape as x.

        edges          centers          first           last
    e - + - + - e   + - + - + - +   + - + - + - +   + - + - + - +
    | . | . | . |   | c | . | c |   | f | . | . |   | . | . | . |
    + _ + _ + _ +   + _ + _ + _ +   + _ + _ + _ +   + _ + _ + _ +
    | . | . | . |   | . | . | . |   | . | . | . |   | . | . | . |
    + _ + _ + _ +   + _ + _ + _ +   + _ + _ + _ +   + _ + _ + _ +
    | . | . | . |   | c | . | c |   | . | . | . |   | . | . | l |
    e _ + _ + _ e   + _ + _ + _ +   + _ + _ + _ +   + _ + _ + _ +

Parameters
----------
image : (batch, channel, *inshape) tensor
    Image to resize
factor : float or list[float], optional
    Resizing factor
    * > 1 : larger image <-> smaller voxels
    * < 1 : smaller image <-> larger voxels
shape : (ndim,) list[int], optional
    Output shape
anchor : {'centers', 'edges', 'first', 'last'} or list, default='centers'
    * In cases 'c' and 'e', the volume shape is multiplied by the
      zoom factor (and eventually truncated), and two anchor points
      are used to determine the voxel size.
    * In cases 'f' and 'l', a single anchor point is used so that
      the voxel size is exactly divided by the zoom factor.
      This case with an integer factor corresponds to subslicing
      the volume (e.g., `vol[::f, ::f, ::f]`).
    * A list of anchors (one per dimension) can also be provided.
interpolation : int or sequence[int], default=1
    Interpolation order.
prefilter : bool, default=True
    Apply spline pre-filter (= interpolates the input)

Returns
-------
resized : (batch, channel, *shape) tensor
    Resized image

"""
```

### Spline analytical energies

```python
interpol.flowreg(
    flow,
    order,
    absolute=0,
    membrane=0,
    bending=0,
    div=0,
    shears=0,
    voxel_size=None,
    norm=False,
    bound='circulant',
    keepdim=False
)
"""
Compute the regularization of a flow field

Parameters
----------
flow : (*batch, *spatial, ndim) tensor
    Spline coefficients of a flow field
order : int
    Spline order
absolute : float, default=0
    Penalty on absolute displacement
membrane : float, default=0
    Penalty on first derivatives
bending : float, default=0
    Penalty on second derivatives
div : float, default=0
    Penalty on volume changes
shears : float, default=0
    Penalty on shears
voxel_size : [sequence of] float
    Voxel size
norm : bool, default=False
    If True, compute the average energy across the field of view.
    Otherwise, compute the sum (integral) of the energy across the FOV.
bound : bound_like
    Boundary conditions
keepdim : bool, default=False
    Keep reduced dimensions

Returns
-------
convflow : (*batch) tensor
    Flow energy (per batch element)

"""
```

```python
interpol.flowmom(
    flow,
    order,
    absolute=0,
    membrane=0,
    bending=0,
    div=0,
    shears=0,
    voxel_size=None,
    norm=False,
    bound='circulant'
)
"""
Compute the matrix-vector part of the regularization of a flow field

Parameters
----------
flow : (*batch, *spatial, ndim) tensor
    Spline coefficients of a flow field
order : int
    Spline order
absolute : float, default=0
    Penalty on absolute displacement
membrane : float, default=0
    Penalty on first derivatives
bending : float, default=0
    Penalty on second derivatives
div : float, default=0
    Penalty on volume changes
shears : float, default=0
    Penalty on shears
voxel_size : [sequence of] float
    Voxel size
norm : bool or int, default=False
    If True, compute the average energy across the field of view.
    Otherwise, compute the sum (integral) of the energy across the FOV.
bound : bound_like
    Boundary conditions

Returns
-------
convflow : (*batch, *spatial, ndim) tensor
    Flow momentum

"""
```

```python
interpol.flowimom(
    mom,
    order,
    absolute=0,
    membrane=0,
    bending=0,
    div=0,
    shears=0,
    voxel_size=None,
    norm=False,
    bound='circulant'
)
"""
Compute the matrix-vector product with the inverse of the reg matrix

!!! warning "Requires `torch >= 1.8`"

Parameters
----------
mom : (*batch, *spatial, ndim) tensor
    Flow momentum
order : int
    Spline order
absolute : float, default=0
    Penalty on absolute displacement
membrane : float, default=0
    Penalty on first derivatives
bending : float, default=0
    Penalty on second derivatives
div : float, default=0
    Penalty on volume changes
shears : float, default=0
    Penalty on shears
voxel_size : [sequence of] float
    Voxel size
norm : bool or int, default=False
    If True, compute the average energy across the field of view.
    Otherwise, compute the sum (integral) of the energy across the FOV.
bound : bound_like
    Boundary conditions

Returns
-------
flow : (*batch, *spatial, ndim) tensor
    Flow field

"""
```

```python
interpol.flow_upsample2(coeff, order, bound='circulant')
"""
Upsample spline coefficients of a flow by a factor 2, while
minimizing the continuous mean squared error.

!!! warning "Requires `torch >= 1.8`"

Parameters
----------
coeff : (*batch, *spatial, ndim) tensor
    Spline coefficients of the flow
order : [list of] int
    Spline order
bound : [list of] bound_like
    Boundary condition.

Returns
-------
upcoeff : tensor
    Upsampled spline coefficients of the flow
    Values are also multiplied by 2.
"""
```

```python
interpol.coeff_upsample2(coeff, order, ndim=1, bound='circulant')
"""
Upsample spline coefficients by a factor 2, while minimizing the
continuous MSE

!!! warning "Requires `torch >= 1.8`"

Parameters
----------
coeff : tensor
    Spline coefficients
order : [list of] int
    Spline order
ndim : int
    The last `ndim` dimension(s) are upsampled
bound : [list of] bound_like
    Boundary condition.

Returns
-------
upcoeff : tensor
    Upsampled spline coefficients
"""
```

```python
interpol.make_kernel(
    ndim,
    order,
    absolute=0,
    membrane=0,
    bending=0,
    div=0,
    shears=0,
    voxel_size=None,
    norm=False,
    *,
    kernels1d=None,
    dtype=None,
    device=None
)
r"""
Generate a convolution kernel for a mixture of energies.

The energy of `v` can be computed via `0.5 * (v * conv(v, kernel)).sum()`.

Parameters
----------
ndim : int
    Number of spatial dimensions
order : int or 'fd'
    Spline order. If 'fd', use finite-differences.
absolute : float, default=0
    Penalty on absolute displacement
membrane : float, default=0
    Penalty on first derivatives
bending : float, default=0
    Penalty on second derivatives
div : float, default=0
    Penalty on volume changes
shears : float, default=0
    Penalty on shears
voxel_size : [sequence of] float
    Voxel size
norm : int, default=False
    If $\ge 0$, contains the number of voxels, and the average
    energy across the field of view is computed.
    Otherwise, compute the sum (integral) of the energy across the FOV.

Other Parameters
----------------
kernels1d : list[tensor]
    Precomputed 1d auto-correlations
dtype : torch.dtype, default=torch.float64
    Data type
device : torch.device
    Device (cpu or cuda device)

Returns
-------
kernel : ([ndim, ndim], [1+2*order] * ndim) tensor
    A `[ndim, ndim]` matrix of ND kernels (if `shears > 0` or `div > 0`)
    or `ndim` ND kernels if (`voxel_size` is a list)
    or a single ND kernel (if `voxel_size` is a scalar).
"""
```

## License

torch-interpol is released under the MIT license.
