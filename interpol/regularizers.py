__all__ = [
    'flowreg',
    'flowmom',
    'flowconv',
    'make_kernel',
    'make_absolute_kernel',
    'make_membrane_kernel',
    'make_bending_kernel',
    'make_div_kernel',
    'make_shears_kernel',
]
import torch
from torch.nn import functional as F
from .padding import pad, make_vector, ensure_shape
from .bounds import to_fourier
from .utils import make_list
from .jit_utils import movedim1


BOUND_DEFAULT = 'circulant'


def flowconv(flow, kernel, bound=BOUND_DEFAULT):
    """
    Convolution of a flow field with an odd-sized kernel, with `padding='same'`

    Parameters
    ----------
    flow : (*batch, *spatial, ndim) tensor
        Spline coefficients of a flow field
    kernel : ([[ndim], ndim], *kernelsize) tensor
        Kernel. Kernels can be:

        * "scaled",     when `shape=kernelsize`
        * "diagonal",   when `shape=(ndim, *kernelsize)`
        * "full",       when `shape=(ndim, ndim, *kernelsize)`
    bound : bound_like
        Boundary conditions

    Returns
    -------
    mom : (*batch, *spatial, ndim) tensor
        Flow momentum

    """
    kernel = kernel.to(flow)
    ndim = flow.shape[-1]
    if ndim > 3:
        raise ValueError('Cannot perform ND convolutions when N > 3')
    batch, spatial = flow.shape[:-ndim-1], flow.shape[-ndim-1:-1]
    kernelsize = kernel.shape[-ndim:]

    # pad
    padsize = [(k-1)//2 for k in kernelsize]
    newsize = [s + 2*p for s, p in zip(spatial, padsize)]
    flow = _padwrapper(flow, padsize, bound)

    # convolve
    batch1 = batch.numel() if batch else 1
    flow = movedim1(flow.reshape([batch1, *newsize, ndim]), -1, 1)
    convnd = getattr(torch.nn.functional, f'conv{ndim}d')
    groups = ndim
    if kernel.ndim == ndim + 2:
        groups = 1
    elif kernel.ndim == ndim + 1:
        kernel = kernel.unsqueeze(1)
    elif kernel.ndim == ndim:
        kernel = kernel.expand([ndim, 1, *kernelsize])
    else:
        raise ValueError('Kernel should have D, D+1 or D+2 dimensions')
    flow = convnd(flow, kernel, groups=groups)

    # return
    flow = movedim1(flow, 1, -1).reshape([*batch, *spatial, ndim])
    return flow


def _padwrapper(flow, padsize, bound):
    # use torch padding when possible
    ndim = flow.shape[-1]
    bound = make_list(bound, ndim)
    bound = list(map(to_fourier, bound))

    use_torch = False
    if len(set(bound)) == 1:
        if bound in ('replicate', 'zero', 'dct1', 'dft'):
            if flow.shape[-1] <= 3:
                if flow.ndim <= flow.shape[-1] + 2:
                    use_torch = True

    flow = movedim1(flow, -1, -ndim-1)
    if use_torch:
        padsize = [p for pp in zip(padsize, padsize) for p in pp]
        bound = dict(
            replicate='border',
            zero='constant',
            dct1='reflect',
            dft='circular',
        )[bound[0]]
        return F.pad(flow, padsize, bound)
    else:
        flow = pad(flow, padsize, bound, side='both')

    return movedim1(flow, -ndim-1, -1)


def flowreg(flow, order, *args, bound=BOUND_DEFAULT, keepdim=False, **kwargs):
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
    norm : bool, default=False
        If True, compute the average energy across the field of view.
        Otherwise, compute the sum (integral) of the energy across the FOV.
    bound : bound_like, default='neumann'
        Boundary conditions
    keepdim : bool, default=False
        Keep reduced dimensions

    Returns
    -------
    convflow : (*batch) tensor
        Flow energy (per batch element)

    """
    ndim = flow.shape[-1]
    mom = flowmom(flow, order, *args, bound=bound, **kwargs)
    return (flow * mom).sum(list(range(-ndim-1, 0)), keepdim=keepdim).mul_(0.5)


def flowmom(flow, order, *args, bound=BOUND_DEFAULT, **kwargs):
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
    norm : bool or int, default=False
        If True, compute the average energy across the field of view.
        Otherwise, compute the sum (integral) of the energy across the FOV.
    bound : bound_like, default='neumann'
        Boundary conditions

    Returns
    -------
    convflow : (*batch, *spatial, ndim) tensor
        Flow momentum

    """
    ndim = flow.shape[-1]
    norm = kwargs.pop('norm', 0)
    if norm:
        kwargs['norm'] = flow.shape[-ndim-1:-1].numel()
    kernel = make_kernel(ndim, order, *args, **kwargs)
    mom = flowconv(flow, kernel, bound=bound)
    # if norm:
    #     mom /= mom.shape[-ndim-1:-1].numel()
    return mom


# ======================================================================
# Analytical autocorrelations [b * b], [b' * b'], [b" * b"] at 0, 1, 2...
# ======================================================================


def make_corrs_order0():
    FF = [1]
    GG = [0]
    HH = [0]
    FG = [0]
    return FF, GG, HH, FG


def make_corrs_order1():
    FF = [2/3, 1/6]
    GG = [2, -1]
    HH = [0, 0]
    FG = [0, 1/2]
    return FF, GG, HH, FG


def make_corrs_order2():
    FF = [11/20, 13/60, 1/120]
    GG = [1, -1/3, -1/6]
    HH = [6, -4, -1]
    FG = [0, 5/12, 1/24]
    return FF, GG, HH, FG


def make_corrs_order3():
    FF = [151/315, 397/1680, 1/42, 1/5040]
    GG = [2/3, -1/8,  -1/5,  -1/120]
    HH = [8/3, -3/2, 0, 1/6]
    FG = [0, 49/144, 7/90,  1/720]
    return FF, GG, HH, FG


def make_corrs_order4():
    FF = [15619/36288, 44117/181440, 913/22680,  2083/362880, 1/362880]
    GG = [35/72, -11/360, -17/90, -187/5040, -1/5040]
    HH = [19/12, -43/60, -4/15, 1/40, 1/120]
    FG = [0, 809/2880, 289/2880, -323/20160, 1/40320]
    return FF, GG, HH, FG


def make_corrs_order5():
    FF = [655177/1663200, 1623019/6652800, 1093/19800,
          50879/13305600, 509/9979200, 1/39916800]
    GG = [809/2160, 1/64, -31/189, -907/24192, -25/18144, -1/362880]
    HH = [31/30, -43/120, -34/105, 239/1680, 29/1260, 1/5040]
    FG = [0, 6787/28800, 16973/151200, 5203/403200, 253/907200, 1/3628800]
    return FF, GG, HH, FG


def make_corrs_order6():
    FF = [27085381/74131200, 125468459/518918400, 28218769/415134720,
          910669/124540416, 1020133/3113510400, 1363/1037836800, 1/6227020800]
    GG = [4319/14400, 11731/302400, -6647/48384, -3455/72576,
          -81313/19958400, -113/2217600, -1/39916800]
    HH = [3101/4320, -1807/10080, -823/2688, 3281/36288,
          169/5670, 83/60480, 1/362880]
    FG = [0, 728741/3628800, 1700933/14515200, 441337/21772800,
          1049/2956800, 2041/239500800, 1/479001600]
    return FF, GG, HH, FG


def make_corrs_order7():
    FF = [2330931341/6810804000, 103795866137/435891456000,
          6423562433/81729648000, 15041229521/1307674368000,
          26502841/40864824000, 13824739/1307674368000,
          2047/81729648000, 1/1307674368000]
    GG = [104159/2073600, 56057/226800, 104159/2073600, -43993/388800,
          -333361/6220800, -14623/2138400, -16081/68428800, -73/55598400,
          -1/6227020800]
    HH = [9871/18900, -102773/1209600, -60317/226800, 166531/3628800,
          50161/1247400, 144499/39916800, 127/2494800, 1/39916800]
    FG = [0, 35263201/203212800, 4489301/38102400, 5532241/203212800,
          233021/104781600, 6323/121927680, 31/165110400, 1/8717829120]
    return FF, GG, HH, FG


corrs = [
    make_corrs_order0(),
    make_corrs_order1(),
    make_corrs_order2(),
    make_corrs_order3(),
    make_corrs_order4(),
    make_corrs_order5(),
    make_corrs_order6(),
    make_corrs_order7(),
]


def make_kernels1d(order, dtype=torch.double, device=None):
    """Build 1D (auto)-correlation kernels"""
    FF, GG, HH, FG = corrs[order]
    FF = FF[1:][::-1] + FF
    GG = GG[1:][::-1] + GG
    HH = HH[1:][::-1] + HH
    FG = list(map(lambda x: -x, FG[1:][::-1])) + FG
    FF = torch.as_tensor(FF, dtype=dtype, device=device)
    GG = torch.as_tensor(GG, dtype=dtype, device=device)
    HH = torch.as_tensor(HH, dtype=dtype, device=device)
    FG = torch.as_tensor(FG, dtype=dtype, device=device)
    return FF, GG, HH, FG


# ======================================================================
# Spline coefficients b, b', b" at 0, 1, 2...
# ======================================================================


def make_evalcoeffs_order0():
    F = [1]
    G = [0]
    H = [0]
    return F, G, H


def make_evalcoeffs_order1():
    F = [1]
    G = [-1]        # not accurate at node centers
    H = [0]
    return F, G, H


def make_evalcoeffs_order2():
    F = [3/4, 1/8]
    G = [0, -1/2]
    H = [-2, 1]     # not accurate at node centers
    return F, G, H


def make_evalcoeffs_order3():
    F = [2/3, 1/6]
    G = [0, -1/2]
    H = [-2, 1]
    return F, G, H


def make_evalcoeffs_order4():
    F = [115/192, 19/96, 1/384]
    G = [0, -11/24, -1/48]
    H = [-5/4, 1/2, 1/8]
    return F, G, H


def make_evalcoeffs_order5():
    F = [11/20, 13/60, 1/120]
    G = [0, -5/12, -1/24]
    H = [-1, 1/3, 1/6]
    return F, G, H


def make_evalcoeffs_order6():
    F = [5887/11520, 10543/46080, 361/23040, 1/46080]
    G = [0, -289/768, -59/960, -1/3840]
    H = [-77/96, 79/384, 37/192, 1/384]
    return F, G, H


def make_evalcoeffs_order7():
    F = [151/315, 397/1680, 1/42, 1/5040]
    G = [0, -49/144, -7/90, -1/720]
    H = [-2/3, 1/8, 1/5, 1/120]
    return F, G, H


evalcoeffs = [
    make_evalcoeffs_order0(),
    make_evalcoeffs_order1(),
    make_evalcoeffs_order2(),
    make_evalcoeffs_order3(),
    make_evalcoeffs_order4(),
    make_evalcoeffs_order5(),
    make_evalcoeffs_order6(),
    make_evalcoeffs_order7(),
]


def make_splinekernels1d(order, dtype=torch.double, device=None):
    """Build 1D spline kernels"""
    F, G, H = evalcoeffs[order]
    F = F[1:][::-1] + F
    H = H[1:][::-1] + H
    G = list(map(lambda x: -x, G[1:][::-1])) + G
    F = torch.as_tensor(F, dtype=dtype, device=device)
    G = torch.as_tensor(G, dtype=dtype, device=device)
    H = torch.as_tensor(H, dtype=dtype, device=device)
    return F, G, H


# ======================================================================
# Discrete autocorrelations [b * b], [b' * b'], [b" * b"] at 0, 1, 2...
# ======================================================================


def make_evalcorrs_order0():
    FF = [1]
    GG = [0]
    HH = [0]
    FG = [0]
    return FF, GG, HH, FG


def make_evalcorrs_order1():
    FF = [1]
    GG = [1]
    HH = [0]
    FG = [-1]
    return FF, GG, HH, FG


def make_evalcorrs_order2():
    FF = [19/32, 3/16, 1/64]
    GG = [1/2, 0, -1/4]
    HH = [6, -4, -1]
    FG = [0, -3/8, -1/16]
    return FF, GG, HH, FG


def make_evalcorrs_order3():
    FF = [1/2, 2/9, 1/36]
    GG = [1/2, 0, -1/4]
    HH = [6, -4, 1]
    FG = [0, -1/3, -1/12]
    return FF, GG, HH, FG


def make_evalcorrs_order4():
    FF = [32227/73728, 1463/6144, 1559/36864, 19/18432, 1/147456]
    GG = [485/1152, 11/576, -121/576, -11/576, -1/2304]
    HH = [67/32, -9/8, -1/16, 1/8, 1/64]
    FG = [0, -2557/9216, -317/3072, -49/9216, -1/18432]
    return FF, GG, HH, FG


def make_evalcorrs_order5():
    FF = [571/1440, 871/3600, 101/1800, 13/3600, 1/14400]
    GG = [101/288, 5/144, -25/144, -5/144, -1/576]
    HH = [23/18, -5/9, -2/9, 1/9, 1/36]
    FG = [0, -169/720, -163/1440, -1/80, -1/2880]
    return FF, GG, HH, FG


def make_evalcorrs_order6():
    FF = [194465143/530841600, 63969833/265420800, 145179247/2123366400,
          1272599/176947200, 18079/70778880, 361/530841600, 1/2123366400]
    GG = [357287/1228800, 14219/307200, -139009/983040, -17051/368640,
          -29293/7372800, -59/1843200, -1/14745600]
    HH = [29575/36864, -4603/18432, -39185/147456, 923/12288, 313/8192,
          37/36864, 1/147456]
    FG = [0, -985339/4915200, -2311229/19660800, -1777493/88473600,
          -9119/8847360, -479/88473600, -1/176947200]
    return FF, GG, HH, FG


def make_evalcorrs_order7():
    FF = [6907/20160, 62927/264600, 666901/8467200, 18167/1587600,
          2797/4233600, 1/105840, 1/25401600]
    GG = [3509/14400, 287/5400, -147/1280, -343/6480, -1813/259200,
          -7/32400, -1/518400]
    HH = [667/1200, -17/150, -239/960, 7/180, 101/2400, 1/300, 1/14400]
    FG = [0, -437/2520, -142679/1209600, -6157/226800, -2039/907200,
          -11/226800, -1/3628800]
    return FF, GG, HH, FG


evalcorrs = [
    make_evalcorrs_order0(),
    make_evalcorrs_order1(),
    make_evalcorrs_order2(),
    make_evalcorrs_order3(),
    make_evalcorrs_order4(),
    make_evalcorrs_order5(),
    make_evalcorrs_order6(),
    make_evalcorrs_order7(),
]


def make_evalkernels1d(order, dtype=torch.double, device=None):
    """Build 1D (auto)-correlation kernels"""
    FF, GG, HH, FG = evalcorrs[order]
    FF = FF[1:][::-1] + FF
    GG = GG[1:][::-1] + GG
    HH = HH[1:][::-1] + HH
    FG = list(map(lambda x: -x, FG[1:][::-1])) + FG
    FF = torch.as_tensor(FF, dtype=dtype, device=device)
    GG = torch.as_tensor(GG, dtype=dtype, device=device)
    HH = torch.as_tensor(HH, dtype=dtype, device=device)
    FG = torch.as_tensor(FG, dtype=dtype, device=device)
    return FF, GG, HH, FG


# ======================================================================
# Build convolution kernels
# ======================================================================


def _make_nd(x, ndim, dim=0):
    """Reshape as a ND tensor, with non-singleton dimension at `dim`"""
    newshape = [1] * ndim
    newshape[dim] = x.numel()
    return x.reshape(newshape)


def make_kernel(ndim, order, absolute=0, membrane=0, bending=0,
                div=0, shears=0, voxel_size=None, norm=False,
                *, kernels1d=None, **backend):
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
    fdiff = order == 'fd'
    if not fdiff:
        kernels1d = kernels1d or make_kernels1d(order, **backend)
    args = (ndim, order, voxel_size, norm)
    kwargs = {'kernels1d': kernels1d, **backend}
    K = 0
    if absolute:
        K += make_absolute_kernel(*args, **kwargs) * absolute
    if membrane:
        if fdiff and torch.is_tensor(K):
            K = ensure_shape(K, [3] * ndim, side='both')
        K += make_membrane_kernel(*args, **kwargs) * membrane
    if bending:
        if fdiff and torch.is_tensor(K):
            K = ensure_shape(K, [5] * ndim, side='both')
        K += make_bending_kernel(*args, **kwargs) * bending
    if shears or div:
        if torch.is_tensor(K):
            slicer = (Ellipsis,)
            if fdiff:
                if bending:
                    slicer += (slice(1, -1),) * ndim
                elif K.shape[-1] < 3:
                    K = ensure_shape(K, [3] * ndim, side='both')
            K0 = K
            K = K0.new_zeros([ndim, ndim] + K0.shape[-ndim:])
            movedim1(K.diagonal(0, 0, 1), -1, 0).copy_(K0)
            if div:
                K[slicer] += make_div_kernel(*args, **kwargs) * div
            if shears:
                K[slicer] += make_shears_kernel(*args, **kwargs) * shears
        else:
            if div:
                K += make_div_kernel(*args, **kwargs) * div
            if shears:
                K += make_shears_kernel(*args, **kwargs) * shears
    return K


def make_absolute_kernel(ndim, order, voxel_size=None, norm=False,
                         *, kernels1d=None, **backend):
    r"""
    Generate a convolution kernel for the Absolute energy.

    The energy of `v` can be computed via `0.5 * (v * conv(v, kernel)).sum()`.

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions
    order : int or 'fd'
        Spline order. If 'fd', use finite-differences.
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
    kernel : ([ndim], [1+2*order] * ndim) tensor
        `ndim` ND kernels (if `voxel_size` is a list) or
        a single ND kernel (if `voxel_size` is a scalar).
    """
    if order == 'fd':
        order = 0
    FF, *_ = kernels1d or make_kernels1d(order, **backend)
    FF = FF.to(**backend)
    # build kernel
    K = 1
    for d in range(ndim):
        K = K * _make_nd(FF, ndim, d)
    # voxel size scaling
    if hasattr(voxel_size, '__iter__'):
        voxel_size = make_vector(voxel_size, ndim,
                                 dtype=FF.dtype, device=FF.device)
        K = _make_nd(voxel_size, ndim+1).square() * K
        if not norm:
            K *= voxel_size.prod()
    elif voxel_size:
        K = K * (voxel_size ** (2 + ndim * (not norm)))
    if norm:
        K /= norm
    return K


def make_membrane_kernel(ndim, order, voxel_size=None, norm=False,
                         *, kernels1d=None, **backend):
    r"""
    Generate a convolution kernel for the Membrane energy.

    The energy of `v` can be computed via `0.5 * (v * conv(v, kernel)).sum()`.

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions
    order : int or 'fd'
        Spline order. If 'fd', use finite-differences.
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
    kernel : ([ndim], [1+2*order] * ndim) tensor
        `ndim` ND kernels (if `voxel_size` is a list)
        or a single ND kernel (if `voxel_size` is a scalar).
    """
    if order == 'fd':
        return make_membrane_kernel_fd(ndim, voxel_size, norm, **backend)
    FF, GG, *_ = kernels1d or make_kernels1d(order, **backend)
    FF, GG = FF.to(**backend), GG.to(**backend)
    vx = voxel_size
    # voxel size scaling
    if hasattr(vx, '__iter__'):
        vx = make_vector(vx, ndim, dtype=FF.dtype, device=FF.device)
        vx2 = vx.square()
        ivx2 = vx2.reciprocal()
    # build kernel
    K = 0
    for i in range(ndim):
        Ki = 1
        for j in range(ndim):
            Ki = Ki * _make_nd(GG if i == j else FF, ndim, j)
        if hasattr(voxel_size, '__iter__'):
            Ki = Ki * ivx2[j]
        K += Ki
    # voxel size scaling
    if hasattr(vx, '__iter__'):
        K = _make_nd(vx2, ndim+1) * K
        if not norm:
            K *= vx.prod()
    elif vx and not norm:
        K = K * (vx ** ndim)
    if norm:
        K /= norm
    return K


def make_bending_kernel(ndim, order, voxel_size=None, norm=False,
                        *, kernels1d=None, **backend):
    r"""
    Generate a convolution kernel for the Bending energy.

    The energy of `v` can be computed via `0.5 * (v * conv(v, kernel)).sum()`.

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions
    order : int or 'fd'
        Spline order. If 'fd', use finite-differences.
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
    kernel : ([ndim], [1+2*order] * ndim) tensor
        `ndim` ND kernels (if `voxel_size` is a list)
        or a single ND kernel (if `voxel_size` is a scalar).
    """
    if order == 'fd':
        return make_bending_kernel_fd(ndim, voxel_size, norm, **backend)
    FF, GG, HH, _ = kernels1d or make_kernels1d(order, **backend)
    FF, GG, HH = FF.to(**backend), GG.to(**backend), HH.to(**backend)
    vx = voxel_size
    # voxel size scaling
    if hasattr(vx, '__iter__'):
        vx = make_vector(vx, ndim, dtype=FF.dtype, device=FF.device)
        vx2 = vx.square()
        ivx2 = vx2.reciprocal()
    # build kernel
    K = 0
    for i in range(ndim):
        for j in range(i, ndim):
            Kij = 1
            for d in range(ndim):
                Kij = Kij * _make_nd(HH if i == j == d else
                                     GG if d in (i, j) else
                                     FF, ndim, d)
            if hasattr(voxel_size, '__iter__'):
                Kij = Kij * (ivx2[i] * ivx2[j])
            if i != j:
                Kij *= 2
            K += Kij
    # voxel size scaling
    if hasattr(vx, '__iter__'):
        K = _make_nd(vx2, ndim+1) * K
        if not norm:
            K *= vx.prod()
    elif vx:
        K = K * (vx ** (ndim * (not norm) - 2))
    if norm:
        K /= norm
    return K


def make_div_kernel(ndim, order, voxel_size=None, norm=False,
                    *, kernels1d=None, **backend):
    r"""
    Generate a convolution kernel for the divergence part of the
    Linear-Elastic energy (lambda).

    The energy of `v` can be computed via `0.5 * (v * conv(v, kernel)).sum()`.

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions
    order : int or 'fd'
        Spline order. If 'fd', use finite-differences.
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
    kernel : ([ndim], [1+2*order] * ndim) tensor
        `ndim` ND kernels (if `voxel_size` is a list)
        or a single ND kernel (if `voxel_size` is a scalar).
    """
    if order == 'fd':
        return make_div_kernel_fd(ndim, voxel_size, norm, **backend)
    FF, GG, _, FG = kernels1d or make_kernels1d(order, **backend)
    FF, GG, FG = FF.to(**backend), GG.to(**backend), FG.to(**backend)
    vx = voxel_size
    # build kernel
    K = FF.new_empty([ndim, ndim] + [len(FF)] * ndim)
    for i in range(ndim):
        Kii = 1
        for d in range(ndim):
            Kii = Kii * _make_nd(GG if i == d else FF, ndim, d)
        K[i, i] = Kii
        for j in range(i+1, ndim):
            Kij = -1
            for d in range(ndim):
                Kij = Kij * _make_nd(FG if d in (i, j) else FF, ndim, d)
            K[i, j] = K[j, i] = Kij
    # voxel size scaling
    vx = voxel_size
    if hasattr(vx, '__iter__'):
        vx = make_vector(vx, ndim, dtype=FF.dtype, device=FF.device)
        if not norm:
            K *= vx.prod()
    elif vx and not norm:
        K = K * (vx ** ndim)
    if norm:
        K /= norm
    return K


def make_shears_kernel(ndim, order, voxel_size=None, norm=False,
                       *, kernels1d=None, **backend):
    r"""
    Generate a convolution kernel for the shears part of the
    Linear-Elastic energy (mu).

    The energy of `v` can be computed via `0.5 * (v * conv(v, kernel)).sum()`.

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions
    order : int or 'fd'
        Spline order. If 'fd', use finite-differences.
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
    kernel : ([ndim], [1+2*order] * ndim) tensor
        `ndim` ND kernels (if `voxel_size` is a list)
        or a single ND kernel (if `voxel_size` is a scalar).
    """
    if order != 'fd':
        FF, GG, HH, FG = kernels1d or make_kernels1d(order, **backend)
        FF, GG, FG = FF.to(**backend), GG.to(**backend), FG.to(**backend)
        kernels1d = (FF, GG, HH, FG)
    membrane = make_membrane_kernel(ndim, order, voxel_size, norm,
                                    kernels1d=kernels1d, **backend)
    lame_div = make_div_kernel(ndim, order, voxel_size, norm,
                               kernels1d=kernels1d, **backend)
    K = lame_div
    movedim1(K.diagonal(0, 0, 1), -1, 0).add_(membrane)
    return K


def make_membrane_kernel_fd(ndim, voxel_size=None, norm=False, **backend):
    # FINITE-DIFFERENCE FALLBACK
    backend.setdefault('dtype', torch.float64)
    K = torch.zeros([3] * ndim)
    vx = voxel_size
    if hasattr(vx, '__iter__'):
        vx = make_vector(vx, ndim, dtype=K.dtype, device=K.device)
        vx2 = vx.square()
        ivx2 = vx2.reciprocal()
    # off-diagonal
    idims = list(range(ndim))
    for di, i in zip(idims * 2, [0] * ndim + [2] * ndim):
        slicer = tuple(i if d == di else 1 for d in range(ndim))
        if hasattr(vx, '__iter__'):
            K[slicer] = -1 * ivx2[di]
        else:
            K[slicer] = -1
    # diagonal (leverage kernel sums to zero)
    K[(1,) * ndim] = -K.sum()
    # voxel size scaling
    if hasattr(voxel_size, '__iter__'):
        K = _make_nd(vx2, ndim+1) * K
        if not norm:
            K *= vx.prod()
    elif voxel_size and not norm:
        K = K * (voxel_size ** ndim)
    if norm:
        K /= norm
    return K


def make_bending_kernel_fd(ndim, voxel_size=None, norm=False, **backend):
    # FINITE-DIFFERENCE FALLBACK
    backend.setdefault('dtype', torch.float64)
    K = torch.zeros([5] * ndim)
    vx = voxel_size
    if hasattr(vx, '__iter__'):
        vx = make_vector(vx, ndim, dtype=K.dtype, device=K.device)
        vx2 = vx.square()
        ivx2 = vx2.reciprocal()
    # off-diagonal
    idims = list(range(ndim))
    for di, i in zip(idims * 2, [1] * ndim + [3] * ndim):
        slicer = tuple(i if d == di else 2 for d in range(ndim))
        if hasattr(vx, '__iter__'):
            K[slicer] = -4 * ivx2[di] * ivx2.sum()
        else:
            K[slicer] = -4 * ndim
        jdims = [d for d in range(ndim) if d != di]
        for dj, j in zip(jdims * 2, [1] * (ndim-1) + [3] * (ndim-1)):
            slicer = tuple(i if d == di else j if d == dj else 2
                           for d in range(ndim))
            if hasattr(vx, '__iter__'):
                K[slicer] = 2 * ivx2[di] * ivx2[dj]
            else:
                K[slicer] = 2
    for di, i in zip(idims * 2, [0] * ndim + [4] * ndim):
        slicer = tuple(i if d == di else 2 for d in range(ndim))
        if hasattr(vx, '__iter__'):
            K[slicer] = ivx2[di] * ivx2[di]
        else:
            K[slicer] = 1
    # diagonal (leverage kernel sums to zero)
    K[(2,) * ndim] = -K.sum()
    # voxel size scaling
    if hasattr(vx, '__iter__'):
        K = _make_nd(vx2, ndim+1) * K
        if not norm:
            K *= vx.prod()
    elif vx:
        K = K * (vx ** (ndim * (int(norm) == 0) - 2))
    if norm:
        K /= norm
    return K


def make_div_kernel_fd(ndim, voxel_size=None, norm=False, **backend):
    # FINITE-DIFFERENCE FALLBACK
    backend.setdefault('dtype', torch.float64)
    K = torch.zeros([ndim, ndim] + [3] * ndim)
    for d in range(ndim):
        # diagonal
        K[(d, d) + (1,) * ndim] = 2
        # off-diagonal
        idims = list(range(ndim))
        for i in [0, 2]:
            slicer = tuple(i if d == dd else 1 for dd in range(ndim))
            K[(d, d) + slicer] = -1
    # off-off-diagonal
    idims = list(range(ndim))
    for di, i in zip(idims * 2, [0] * ndim + [2] * ndim):
        jdims = [d for d in range(ndim) if d != di]
        for dj, j in zip(jdims * 2, [0] * (ndim - 1) + [2] * (ndim - 1)):
            slicer = tuple(i if d == di else j if d == dj else 1
                           for d in range(ndim))
            K[(di, dj) + slicer] = -0.25 if i == j else 0.25
    # voxel size scaling
    vx = voxel_size
    if hasattr(voxel_size, '__iter__'):
        vx = make_vector(vx, ndim, dtype=K.dtype, device=K.device)
        if not norm:
            K *= vx.prod()
    elif voxel_size and not norm:
        K = K * (vx ** ndim)
    if norm:
        K /= norm
    return K
