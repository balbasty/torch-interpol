"""
# Boundary conditions

There is no common convention to name boundary conditions.
This file lists all possible aliases and provides tool to "convert"
between them. It also defines function that can be used to implement
these boundary conditions.

| Fourier                                     | SciPy `ndimage`          | Numpy `pad`   | PyTorch `pad` | PyTorch `grid_sample`            | Other                   | Description               |
| ------------------------------------------- | ------------------------ | ------------- | ------------- | -------------------------------- | ----------------------- | ------------------------- |
| [`replicate`](anyreg.core.bounds.replicate) | nearest                  | edge          | border        | replicate                        | repeat                  | ` a  a | a b c d |  d  d` |
| `zero`                                      | constant, grid-constant  | constant      | constant      | zeros                            |                         | ` 0  0 | a b c d |  0  0` |
| [`dct1`](anyreg.core.bounds.dct1)           | mirror                   | reflect       | reflect       | reflection (align_corners=False) |                         | ` c  b | a b c d |  c  b` |
| [`dct2`](anyreg.core.bounds.dct2)           | reflect, grid-mirror     | symmetric     |               | reflection (align_corners=True)  | neumann                 | ` b  a | a b c d |  d  c` |
| [`dst1`](anyreg.core.bounds.dst1)           |                          |               |               |                                  | antimirror              | `-a  0 | a b c d |  0 -d` |
| [`dst2`](anyreg.core.bounds.dst2)           |                          |               |               |                                  | antireflect, dirichlet  | `-b -a | a b c d | -d -c` |
| [`dft`](anyreg.core.bounds.dft)             | grid-wrap                | wrap          | circular      |                                  | circulant               | ` c  d | a b c d |  a  b` |
|                                             | wrap                     |               |               |                                  |                         | ` c  d | a b c d |  b  c` |
|                                             |                          | linear_ramp   |
|                                             |                          | minimum, maximum, mean, median |

Some of these conventions are inconsistant with each other. For example
`"wrap"` in `scipy.ndimage` is different from `"wrap"` in `numpy.pad`,
which corresponds to `"grid-wrap"` in `scipy.ndimage`. Also, `"reflect"`
in `numpy.pad` and `torch.pad` is different from `"reflect"` in `scipy.ndimage`,
which correspond to `"symmetric"` in `numpy.pad`. Because of these
consistencies, in `interpol`, `"wrap"` and `"reflect"` are respectively
synonymous with `"dft"` and `"dct2"`.

Classes
-------
BoundType
    Enum type for bounds
ExtrapolateType
    Enum type for extrapolations
Bound
    Just-in-time index wrapper

Functions
---------
to_enum
    Convert boundary type to `BoundType`
to_int
    Convert boundary type to `BoundType`-based integer values
to_fourier
    Convert boundary type to discrete transforms
to_scipy
    Convert boundary type to scipy convention
to_torch
    Convert boundary type to `torch.grid_sample` convention
replicate
    Apply replicate (nearest/border) boundary conditions to an index (not jit).
    **Aliases:**
    [`border`][interpol.bounds.border],
    [`nearest`][interpol.bounds.nearest],
    [`repeat`][interpol.bounds.repeat].
dft
    Apply DFT (circulant/wrap) boundary conditions to an index (not jit).
    **Aliases:**
    [`wrap`][interpol.bounds.wrap],
    [`circular`][interpol.bounds.circular],
    [`circulant`][interpol.bounds.circulant].
dct1
    Apply DCT-I (mirror) boundary conditions to an index (not jit).
    **Alias:** [`mirror`][interpol.bounds.mirror]
dct2
    Apply DCT-II (reflect) boundary conditions to an index (not jit).
    **Aliases**:
    [`reflect`][interpol.bounds.reflect],
    [`neumann`][interpol.bounds.neumann].
dst1
    Apply DST-I (antimirror) boundary conditions to an index (not jit).
    **Alias:** [`antimirror`][interpol.bounds.antimirror]
dst2
    Apply DST-II (antireflect/wrap) boundary conditions to an index (not jit).
    **Aliases**:
    [`antireflect`][interpol.bounds.antireflect],
    [`dirichlet`][interpol.bounds.dirichlet].

"""  # noqa: E501

import torch
from torch import Tensor
from enum import Enum
from typing import Optional, Tuple
from .jit_utils import floor_div


class BoundType(Enum):
    zero = zeros = constant = 0
    replicate = repeat = nearest = border = edge = 1
    dct1 = mirror = 2
    dct2 = reflect = reflection = neumann = 3
    dst1 = antimirror = 4
    dst2 = antireflect = dirichlet = 5
    dft = wrap = circular = circulant = 6
    nocheck = -1


class ExtrapolateType(Enum):
    no = 0     # threshold: (0, n-1)
    yes = 1
    hist = 2   # threshold: (-0.5, n-0.5)


@torch.jit.script
class Bound:

    def __init__(self, bound_type: int = 3):
        self.type = bound_type

    def index(self, i, n: int):
        if self.type in (0, 1):  # zero / replicate
            return i.clamp(min=0, max=n-1)
        elif self.type in (3, 5):  # dct2 / dst2
            n2 = n * 2
            i = torch.where(i < 0, (-i-1).remainder(n2).neg().add(n2 - 1),
                            i.remainder(n2))
            i = torch.where(i >= n, -i + (n2 - 1), i)
            return i
        elif self.type == 2:  # dct1
            if n == 1:
                return torch.zeros(i.shape, dtype=i.dtype, device=i.device)
            else:
                n2 = (n - 1) * 2
                i = i.abs().remainder(n2)
                i = torch.where(i >= n, -i + n2, i)
                return i
        elif self.type == 4:  # dst1
            n2 = 2 * (n + 1)
            first = torch.zeros([1], dtype=i.dtype, device=i.device)
            last = torch.full([1], n - 1, dtype=i.dtype, device=i.device)
            i = torch.where(i < 0, -i - 2, i)
            i = i.remainder(n2)
            i = torch.where(i > n, -i + (n2 - 2), i)
            i = torch.where(i == -1, first, i)
            i = torch.where(i == n, last, i)
            return i
        elif self.type == 6:  # dft
            return i.remainder(n)
        else:
            return i

    def transform(self, i, n: int) -> Optional[Tensor]:
        if self.type == 4:  # dst1
            if n == 1:
                return None
            one = torch.ones([1], dtype=torch.int8, device=i.device)
            zero = torch.zeros([1], dtype=torch.int8, device=i.device)
            n2 = 2 * (n + 1)
            i = torch.where(i < 0, -i + (n-1), i)
            i = i.remainder(n2)
            x = torch.where(i == 0, zero, one)
            x = torch.where(i.remainder(n + 1) == n, zero, x)
            i = floor_div(i, n+1)
            x = torch.where(torch.remainder(i, 2) > 0, -x, x)
            return x
        elif self.type == 5:  # dst2
            i = torch.where(i < 0, n - 1 - i, i)
            x = torch.ones([1], dtype=torch.int8, device=i.device)
            i = floor_div(i, n)
            x = torch.where(torch.remainder(i, 2) > 0, -x, x)
            return x
        elif self.type == 0:  # zero
            one = torch.ones([1], dtype=torch.int8, device=i.device)
            zero = torch.zeros([1], dtype=torch.int8, device=i.device)
            outbounds = ((i < 0) | (i >= n))
            x = torch.where(outbounds, zero, one)
            return x
        else:
            return None


bounds_fourier = ('replicate', 'zero', 'dct2', 'dct1', 'dst2', 'dst1', 'dft')
bounds_scipy = ('nearest', 'constant', 'reflect', 'mirror', 'wrap')
bounds_torch = ('nearest', 'zeros', 'reflection')
bounds_torch_pad = ('border', 'constant', 'reflect', 'circular')
bounds_other = ('repeat', 'neumann', 'circular', 'circulant',
                'antireflect', 'dirichlet', 'antimirror')
enum_bounds = (BoundType.zero, BoundType.repeat, BoundType.dct1,
               BoundType.dct2, BoundType.dst1, BoundType.dst2, BoundType.dft)
int_bounds = tuple(range(7))


zero_bounds = [b for b in BoundType.__members__.keys()
               if getattr(BoundType, b) == BoundType.zero]
rept_bounds = [b for b in BoundType.__members__.keys()
               if getattr(BoundType, b) == BoundType.repeat]
dct1_bounds = [b for b in BoundType.__members__.keys()
               if getattr(BoundType, b) == BoundType.dct1]
dct2_bounds = [b for b in BoundType.__members__.keys()
               if getattr(BoundType, b) == BoundType.dct2]
dst1_bounds = [b for b in BoundType.__members__.keys()
               if getattr(BoundType, b) == BoundType.dst1]
dst2_bounds = [b for b in BoundType.__members__.keys()
               if getattr(BoundType, b) == BoundType.dst2]
dft_bounds = [b for b in BoundType.__members__.keys()
              if getattr(BoundType, b) == BoundType.dft]


def to_enum(bound) -> BoundType:
    """Convert boundary type to interpol enum type.

    !!! note "See also"
        * [`to_fourier`][interpol.bounds.to_fourier]
        * [`to_scipy`][interpol.bounds.to_scipy]
        * [`to_torch`][interpol.bounds.to_torch]
        * [`to_int`][interpol.bounds.to_int]

    Parameters
    ----------
    bound : [list of] str
        Boundary condition in any convention

    Returns
    -------
    bound : [list of] BoundType
        Boundary condition

    """
    intype = type(bound)
    if not isinstance(bound, (list, tuple)):
        bound = [bound]
    obound = []
    for b in bound:
        if isinstance(b, str):
            b = b.lower()
        if b in (*zero_bounds, BoundType.zero, 0):
            obound.append(BoundType.zero)
        elif b in (*rept_bounds, BoundType.border, 1):
            obound.append(BoundType.replicate)
        elif b in (*dct1_bounds, BoundType.dct1, 2):
            obound.append(BoundType.dct1)
        elif b in (*dct2_bounds, BoundType.dct2, 3):
            obound.append(BoundType.dct2)
        elif b in (*dst1_bounds, BoundType.dst1, 4):
            obound.append(BoundType.dst1)
        elif b in (*dst2_bounds, BoundType.dst2, 5):
            obound.append(BoundType.dst2)
        elif b in (*dft_bounds, BoundType.dft, 6):
            obound.append(BoundType.dft)
        elif b in ('nocheck', BoundType.nocheck, -1):
            obound.append(BoundType.nocheck)
        else:
            raise ValueError(f'Unknown boundary condition {b}')
    if issubclass(intype, (list, tuple)):
        obound = intype(obound)
    else:
        obound = obound[0]
    return obound


def to_int(bound) -> int:
    """Convert boundary type to interpol enum integer.

    !!! note "See also"
        * [`to_enum`][interpol.bounds.to_enum]
        * [`to_fourier`][interpol.bounds.to_fourier]
        * [`to_scipy`][interpol.bounds.to_scipy]
        * [`to_torch`][interpol.bounds.to_torch]

    Parameters
    ----------
    bound : [list of] str
        Boundary condition in any convention

    Returns
    -------
    bound : [list of] {0..6}
        Boundary condition

    """
    bound = to_enum(bound)
    if isinstance(bound, (list, tuple)):
        bound = type(bound)(map(lambda x: x.value, bound))
    else:
        bound = bound.value
    return bound


def to_fourier(bound):
    """Convert boundary type to discrete transforms.

    !!! note "See also"
        * [`to_enum`][interpol.bounds.to_enum]
        * [`to_scipy`][interpol.bounds.to_scipy]
        * [`to_torch`][interpol.bounds.to_torch]
        * [`to_int`][interpol.bounds.to_int]

    Parameters
    ----------
    bound : [list of] str
        Boundary condition in any convention

    Returns
    -------
    bound : [list of] {'replicate', 'zero', 'dct2', 'dct1', 'dst2', 'dst1', 'dft'}
        Boundary condition in terms of discrete transforms

    """  # noqa: E501
    intype = type(bound)
    if not isinstance(bound, (list, tuple)):
        bound = [bound]
    obound = []
    for b in bound:
        if isinstance(b, str):
            b = b.lower()
        if b in (*zero_bounds, BoundType.zero, 0):
            obound.append('zero')
        elif b in (*rept_bounds, BoundType.border, 1):
            obound.append('replicate')
        elif b in (*dct1_bounds, BoundType.dct1, 2):
            obound.append('dct1')
        elif b in (*dct2_bounds, BoundType.dct2, 3):
            obound.append('dct2')
        elif b in (*dst1_bounds, BoundType.dst1, 4):
            obound.append('dst1')
        elif b in (*dst2_bounds, BoundType.dst2, 5):
            obound.append('dst2')
        elif b in (*dft_bounds, BoundType.dft, 6):
            obound.append('dft')
        elif b in ('nocheck', BoundType.nocheck, -1):
            obound.append('nocheck')
        else:
            raise ValueError(f'Unknown boundary condition {b}')
    if issubclass(intype, (list, tuple)):
        obound = intype(obound)
    else:
        obound = obound[0]
    return obound


def to_scipy(bound):
    """Convert boundary type to SciPy's convention.

    !!! note "See also"
        * [`to_enum`][interpol.bounds.to_enum]
        * [`to_fourier`][interpol.bounds.to_fourier]
        * [`to_torch`][interpol.bounds.to_torch]
        * [`to_int`][interpol.bounds.to_int]

    Parameters
    ----------
    bound : [list of] str
        Boundary condition in any convention

    Returns
    -------
    bound : [list of] {'border', 'constant', 'reflect', 'mirror', 'wrap'}
        Boundary condition in SciPy's convention

    """
    intype = type(bound)
    if not isinstance(bound, (list, tuple)):
        bound = [bound]
    obound = []
    for b in bound:
        if isinstance(b, str):
            b = b.lower()
        if b in (*zero_bounds, BoundType.zero, 0):
            obound.append('constant')
        elif b in (*rept_bounds, BoundType.border, 1):
            obound.append('border')
        elif b in (*dct1_bounds, BoundType.dct1, 2):
            obound.append('mirror')
        elif b in (*dct2_bounds, BoundType.dct2, 3):
            obound.append('reflect')
        elif b in (*dst1_bounds, BoundType.dst1, 4):
            raise ValueError(f'Boundary condition {b} not available in SciPy.')
        elif b in (*dst2_bounds, BoundType.dst2, 5):
            raise ValueError(f'Boundary condition {b} not available in SciPy.')
        elif b in (*dft_bounds, BoundType.dft, 6):
            obound.append('wrap')
        elif b in ('nocheck', BoundType.nocheck, -1):
            raise ValueError(f'Boundary condition {b} not available in SciPy.')
        else:
            raise ValueError(f'Unknown boundary condition {b}')
    if issubclass(intype, (list, tuple)):
        obound = intype(obound)
    else:
        obound = obound[0]
    return obound


def to_torch(bound):
    """Convert boundary type to PyTorch's convention.

    !!! note "See also"
        * [`to_enum`][interpol.bounds.to_enum]
        * [`to_fourier`][interpol.bounds.to_fourier]
        * [`to_scipy`][interpol.bounds.to_scipy]
        * [`to_int`][interpol.bounds.to_int]

    Parameters
    ----------
    bound : [list of] str
        Boundary condition in any convention

    Returns
    -------
    bound : [list of] ({'nearest', 'zero', 'reflection'}, bool)
        The first element is the boundary condition in PyTorchs's
        convention, and the second element is the value of `align_corners`.

    """
    intype = type(bound)
    if not isinstance(bound, (list, tuple)):
        bound = [bound]
    obound = []
    for b in bound:
        if isinstance(b, str):
            b = b.lower()
        if b in (*zero_bounds, BoundType.zero, 0):
            obound.append(('zero', None))
        elif b in (*rept_bounds, BoundType.border, 1):
            obound.append(('nearest', None))
        elif b in (*dct1_bounds, BoundType.dct1, 2):
            obound.append(('reflection', False))
        elif b in (*dct2_bounds, BoundType.dct2, 3):
            obound.append(('reflection', True))
        elif b in (*dst1_bounds, BoundType.dst1, 4):
            raise ValueError(f'Boundary condition {b} not available in Torch.')
        elif b in (*dst2_bounds, BoundType.dst2, 5):
            raise ValueError(f'Boundary condition {b} not available in Torch.')
        elif b in (*dft_bounds, BoundType.dft, 6):
            raise ValueError(f'Boundary condition {b} not available in Torch.')
        elif b in ('nocheck', BoundType.nocheck, -1):
            raise ValueError(f'Boundary condition {b} not available in Torch.')
        else:
            raise ValueError(f'Unknown boundary condition {b}')
    if issubclass(intype, (list, tuple)):
        obound = intype(obound)
    else:
        obound = obound[0]
    return obound


def nocheck(i, n):
    """Assume all indices are inbounds"""
    return i, 1


def replicate(i, n):
    """Apply replicate (nearest/border) boundary conditions to an index

    !!! info "Aliases"
        `border`, `nearest`, `repeat`

    Parameters
    ----------
    i : int or tensor
        Index
    n : int
        Length of the field of view

    Returns
    -------
    i : int or tensor
        Index that falls inside the field of view [0, n-1]
    s : {1, 0, -1}
        Sign of the transformation (always 1 for replicate)

    """
    return (replicate_script(i, n) if torch.is_tensor(i) else
            replicate_int(i, n))


def replicate_int(i, n):
    return min(max(i, 0), n-1), 1


@torch.jit.script
def replicate_script(i, n: int) -> Tuple[Tensor, int]:
    return i.clamp(min=0, max=n-1), 1


def dft(i, n):
    """Apply DFT (circulant/wrap) boundary conditions to an index

    !!! info "Aliases"
        `wrap`, `circular`

    Parameters
    ----------
    i : int or tensor
        Index
    n : int
        Length of the field of view

    Returns
    -------
    i : int or tensor
        Index that falls inside the field of view [0, n-1]
    s : {1, 0, -1}
        Sign of the transformation (always 1 for dft)

    """
    return dft_script(i, n) if torch.is_tensor(i) else dft_int(i, n)


def dft_int(i, n):
    return i % n, 1


@torch.jit.script
def dft_script(i, n: int) -> Tuple[Tensor, int]:
    return i.remainder(n), 1


def dct2(i, n):
    """Apply DCT-II (reflect) boundary conditions to an index

    !!! info "Aliases"
        `reflect`, `neumann`

    Parameters
    ----------
    i : int or tensor
        Index
    n : int
        Length of the field of view

    Returns
    -------
    i : int or tensor
        Index that falls inside the field of view [0, n-1]
    s : {1, 0, -1}
        Sign of the transformation (always 1 for dct2)

    """
    return dct2_script(i, n) if torch.is_tensor(i) else dct2_int(i, n)


def dct2_int(i: int, n: int) -> Tuple[int, int]:
    n2 = n * 2
    i = (n2 - 1) - i if i < 0 else i
    i = i % n2
    i = (n2 - 1) - i if i >= n else i
    return i, 1


@torch.jit.script
def dct2_script(i, n: int) -> Tuple[Tensor, int]:
    n2 = n * 2
    i = torch.where(i < 0, (n2 - 1) - i, i)
    i = i.remainder(n2)
    i = torch.where(i >= n, (n2 - 1) - i, i)
    return i, 1


def dct1(i, n):
    """Apply DCT-I (mirror) boundary conditions to an index

    !!! info "Aliases"
        `mirror`

    Parameters
    ----------
    i : int or tensor
        Index
    n : int
        Length of the field of view

    Returns
    -------
    i : int or tensor
        Index that falls inside the field of view [0, n-1]
    s : {1, 0, -1}
        Sign of the transformation (always 1 for dct1)

    """
    return dct1_script(i, n) if torch.is_tensor(i) else dct1_int(i, n)


def dct1_int(i: int, n: int) -> Tuple[int, int]:
    if n == 1:
        return 0, 1
    n2 = (n - 1) * 2
    i = abs(i) % n2
    i = n2 - i if i >= n else i
    return i, 1


@torch.jit.script
def dct1_script(i, n: int) -> Tuple[Tensor, int]:
    if n == 1:
        return torch.zeros_like(i), 1
    n2 = (n - 1) * 2
    i = i.abs().remainder(n2)
    i = torch.where(i >= n, n2 - i, i)
    return i, 1


def dst1(i, n):
    """Apply DST-I (antimirror) boundary conditions to an index

    !!! info "Aliases"
        `antimirror`

    Parameters
    ----------
    i : int or tensor
        Index
    n : int
        Length of the field of view

    Returns
    -------
    i : int or tensor
        Index that falls inside the field of view [0, n-1]
    s : [tensor of] {1, 0, -1}
        Sign of the transformation

    """
    return dst1_script(i, n) if torch.is_tensor(i) else dst1_int(i, n)


def dst1_int(i: int, n: int) -> Tuple[int, int]:
    n2 = 2 * (n + 1)

    # sign
    ii = (2*n - i) if i < 0 else i
    ii = (ii % n2) % (n + 1)
    x = 0 if ii == n else 1
    x = -x if (i / (n + 1)) % 2 >= 1 else x

    # index
    i = -i - 2 if i < 0 else i
    i = i % n2
    i = (n2 - 2) - i if i > n else i
    i = min(max(i, 0), n-1)
    return i, x


@torch.jit.script
def dst1_script(i, n: int) -> Tuple[Tensor, Tensor]:
    n2 = 2 * (n + 1)

    # sign
    #   zeros
    ii = torch.where(i < 0, 2*n - i, i).remainder(n2).remainder(n + 1)
    x = (ii != n).to(torch.int8)
    #   +/- ones
    x = torch.where((i / (n + 1)).remainder(2) >= 1, -x, x)

    # index
    i = torch.where(i < 0, -2 - i, i)
    i = i.remainder(n2)
    i = torch.where(i > n, (n2 - 2) - i, i)
    i = i.clamp(0, n-1)
    return i, x


def dst2(i, n):
    """Apply DST-II (antireflect) boundary conditions to an index

    !!! info "Aliases"
        `antireflect`, `dirichlet`

    Parameters
    ----------
    i : int or tensor
        Index
    n : int
        Length of the field of view

    Returns
    -------
    i : int or tensor
        Index that falls inside the field of view [0, n-1]
    s : [tensor of] {1, 0, -1}
        Sign of the transformation (always 1 for dct1)

    """
    return dst2_script(i, n) if torch.is_tensor(i) else dst2_int(i, n)


def dst2_int(i: int, n: int) -> Tuple[int, int]:
    x = -1 if (i/n) % 2 >= 1 else 1
    return dct2_int(i, n)[0], x


@torch.jit.script
def dst2_script(i, n: int) -> Tuple[Tensor, Tensor]:
    x = torch.ones([1], dtype=torch.int8, device=i.device)
    x = torch.where((i / n).remainder(2) >= 1, -x, x)
    return dct2_script(i, n)[0], x


nearest = border = repeat = replicate
reflect = neumann = dct2
mirror = dct1
antireflect = dirichlet = dst2
antimirror = dst1
wrap = circular = circulant = circulant = dft
