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

```shell
pip install git+https://github.com/balbasty/torch-interpol

# Or, alternatively
git clone git@github.com:balbasty/torch-interpol.git
pip install ./torch-interpol
```

## Usage

The most useful function is `grid_pull`, which samples an image at a given 
set of coordinates according to some spline order. Here's a small example 
that show how to reslice an image to a different image space:
```python
# we are going to rotate and resample a 32x32 pixels square
import torch, math
import matplotlib.pyplot as plt
from interpol import grid_pull, affine_grid

# generate a "square" phantom image
x = torch.zeros([64, 64])
x[16:48, 16:48] = 1

# build rotation matrix
rot = [[math.cos(math.pi/4), -math.sin(math.pi/4), 0],
       [math.sin(math.pi/4), math.cos(math.pi/4), 0],
       [0, 0, 1]]
center = [[1, 0, -32],
          [0, 1, -32],
          [0, 0, 1]]
rot = torch.as_tensor(rot, dtype=torch.float)
center = torch.as_tensor(center, dtype=torch.float)
full_affine = center.inverse() @ rot @ center 

# build dense field of sampling coordinates
grid = affine_grid(full_affine, [64, 64])

# resample
y1 = grid_pull(x, grid, bound='mirror', interpolation=1)
y3 = grid_pull(x, grid, bound='mirror', interpolation=3, prefilter=True)
y5 = grid_pull(x, grid, bound='mirror', interpolation=5, prefilter=True)

# plot
plt.subplot(1, 4, 1)
plt.imshow(x, vmin=0, vmax=1)
plt.axis('off')
plt.title('original')
plt.subplot(1, 4, 2)
plt.imshow(y1, vmin=0, vmax=1)
plt.axis('off')
plt.title('1st order')
plt.subplot(1, 4, 3)
plt.imshow(y3, vmin=0, vmax=1)
plt.axis('off')
plt.title('3rd order')
plt.subplot(1, 4, 4)
plt.imshow(y5, vmin=0, vmax=1)
plt.axis('off')
plt.title('5th order')
plt.show()
```

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
Splat an image with respect to a deformation field (pull adjoint).

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
## License

torch-interpol is released under the MIT license.
