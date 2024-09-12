# This file contains energy-related functions that require torch.fft
__all__ = [
    'flowiconv',
    'flowimom',
    'flow_upsample2',
    'coeff_upsample2',
]
import torch
import itertools
import math
from torch.nn import functional as F
from torch.fft import fftn, ifftn, fftshift, ifftshift
from bounds import pad, ensure_shape, to_fourier
from bounds import dctn1, dctn2, idctn1, idctn2, dstn1, idstn1, dstn2, idstn2
from .utils import make_list, batchinv
from .jit_utils import movedim1
from .energies_nofft import make_kernel, make_upkernels1d, BOUND_DEFAULT


def flowiconv(mom, kernel, bound=BOUND_DEFAULT):
    """
    Convolution of a vector field with the _inverse_ of an odd-sized kernel,
    with `padding='same'`.

    Uses a frequency transform (FFT, DCT, DST) under the hood, so is only
    implemented with boundary conditions that map to one of these transforms.

    !!! warning "Requires `torch >= 1.8`"

    Parameters
    ----------
    mom : (*batch, *spatial, ndim) tensor
        Spline coefficients of a momentum field
    kernel : ([[ndim], ndim], *kernelsize) tensor
        Kernel. Kernels can be:

        * "scaled",     when `shape=kernelsize`
        * "diagonal",   when `shape=(ndim, *kernelsize)`
        * "full",       when `shape=(ndim, ndim, *kernelsize)`
    bound : [list of] bound_like
        Boundary conditions

    Returns
    -------
    flow : (*batch, *spatial, ndim) tensor
        Flow field

    """
    ndim = mom.shape[-1]
    if kernel.ndim not in (ndim, ndim+1, ndim+2):
        raise ValueError('Unsupported kernel shape')

    bound = to_fourier(make_list(bound, ndim))
    if any(b not in ('dft', 'dct1', 'dct2', 'dst1', 'dst2') for b in bound):
        raise ValueError('Only boundary conditions that correspond to some '
                         'kind of frequency transform are supported.')

    # Transform kernel
    shape = mom.shape[-ndim-1:-1]
    kernel = _anyftn_kernel(kernel, shape, dft=[b == 'dft' for b in bound],
                            inv=True)

    # Transform momentum
    flow = _anyftn(mom, dim=list(range(-ndim-1, -1)), bound=bound)

    # Product or Matrix product
    if kernel.ndim == ndim + 2:
        kernel = movedim1(movedim1(kernel, 0, -1), 0, -1)
        flow = kernel.matmul(flow.unsqueeze(-1)).squeeze(-1)
    elif kernel.ndim == ndim + 1:
        kernel = movedim1(kernel, 0, -1)
        flow = flow * kernel
    else:
        flow = flow * kernel.unsqueeze(-1)

    # Inverse transform flow
    flow = _anyiftn(flow, dim=list(range(-ndim-1, -1)), bound=bound)

    return flow


def flowimom(mom, order, *args, bound=BOUND_DEFAULT, **kwargs):
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
    ndim = mom.shape[-1]
    norm = kwargs.pop('norm', 0)
    if norm:
        kwargs['norm'] = mom.shape[-ndim-1:-1].numel()
    kernel = make_kernel(ndim, order, *args, **kwargs,
                         dtype=mom.dtype, device=mom.device)
    return flowiconv(mom, kernel, bound)


def flow_upsample2(coeff, order, bound=BOUND_DEFAULT):
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
    ndim = coeff.shape[-1]
    coeff = movedim1(coeff, -1, 0)
    coeff = coeff_upsample2(coeff, order, ndim, bound)
    coeff = movedim1(coeff, 0, -1)
    coeff *= 2
    return coeff


def coeff_upsample2(coeff, order, ndim=1, bound=BOUND_DEFAULT):
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
    bound = to_fourier(make_list(bound, ndim))
    order = make_list(order, ndim)
    conv = getattr(F, f'conv{ndim}d')

    # get 1D kernels
    FF, EE, OO, PE, PO = 1, [], [], [], []
    for d, o in enumerate(order):
        FF1, EE1, OO1 = make_upkernels1d(
            o, dtype=coeff.dtype, device=coeff.device)
        PE += [len(OO1) // 2]
        PO += [len(OO1) - PE[-1] - 1]
        for _ in range(ndim-1-d):
            FF1 = FF1[..., None]
            EE1 = EE1[..., None]
            OO1 = OO1[..., None]
        FF = FF * FF1
        EE += [EE1]
        OO += [OO1]

    # permute/reshape
    batch = coeff.shape[:-ndim]
    coeff0 = coeff.reshape((-1, 1) + coeff.shape[-ndim:])
    upshape = coeff0.shape[:-ndim] + tuple(2*s for s in coeff0.shape[-ndim:])
    coeff = coeff0.new_empty(upshape)

    for is_odd in itertools.product([True, False], repeat=ndim):
        # convolve with (smallspline * bigspline)
        #   odd and even voxels have different (inverted) kernels
        kernel = 1
        slicer = []
        padsize = []
        for d in range(ndim):
            kernel = kernel * (OO[d] if is_odd[d] else EE[d])
            padsize.extend((PO[d], PE[d]) if is_odd[d] else (PE[d], PO[d]))
            slicer += [slice(1, None, 2) if is_odd[d] else slice(0, -1, 2)]
        coeff_conv = pad(coeff0, padsize, bound)
        coeff_conv = conv(coeff_conv, kernel[None, None])
        coeff[(Ellipsis, *slicer)] = coeff_conv

    # permute/reshape
    coeff = coeff.reshape(batch + coeff.shape[-ndim:])

    # Fourier inversion
    lastdims = list(range(-ndim, 0))
    FF = _anyftn_kernel(FF, coeff.shape[-ndim:], [b == 'dft' for b in bound])
    coeff = _anyftn(coeff, lastdims, bound)
    coeff = coeff / FF
    coeff = _anyiftn(coeff, lastdims, bound)
    return coeff


# ======================================================================
# Fourier transform helpers
# ======================================================================


def _anyftn(x, dim=-1, bound='dft', shift=False):
    """
    Apply any type of forward frequency transform (FFT, DCT, DST)

    Parameters
    ----------
    x : tensor
        Input tensor
    dim : [list of] int
        Dimension(s) to transform
    bound : [list of] {'dft', 'dct1', 'dct2', 'dst1', 'dst2'}
        Type of frequency transform to apply
    shift : bool
        If True, apply an `ifftshift` before an `fft`.

    Returns
    -------
    y : tensor
        Outpt tensor
    """
    dim = make_list(dim)
    bound = to_fourier(make_list(bound, len(dim)))
    if any(b not in ('dft', 'dct1', 'dct2', 'dst1', 'dst2') for b in bound):
        raise ValueError('Bounds must correspond to a frequency tranform')
    if len(set(bound)) == 1:
        bound = bound[0]
        if bound == 'dft':
            if shift:
                x = ifftshift(x, dim=dim)
            x = fftn(x, dim=dim)
        elif bound == 'dct1':
            x = dctn1(x, dim=dim)
        elif bound == 'dct2':
            x = dctn2(x, dim=dim)
        elif bound == 'dst1':
            x = dstn1(x, dim=dim)
        elif bound == 'dst2':
            x = dstn2(x, dim=dim)
    else:
        for d, b in zip(dim, bound):
            if b == 'dft':
                if shift:
                    x = ifftshift(x, dim=d)
                x = fftn(x, dim=d)
            elif b == 'dct1':
                x = dctn1(x, dim=d)
            elif b == 'dct2':
                x = dctn2(x, dim=d)
            elif b == 'dst1':
                x = dstn1(x, dim=d)
            elif b == 'dst2':
                x = dstn2(x, dim=d)
    return x


def _anyiftn(x, dim=-1, bound='dft', shift=False):
    """
    Apply any type of inverse frequency transform (FFT, DCT, DST)

    Parameters
    ----------
    x : tensor
        Input tensor
    dim : [list of] int
        Dimension(s) to transform
    bound : [list of] {'dft', 'dct1', 'dct2', 'dst1', 'dst2'}
        Type of frequency transform to apply
    shift : bool
        If True, apply an `fftshift` after an `ifft`.

    Returns
    -------
    y : tensor
        Outpt tensor
    """
    dim = make_list(dim)
    bound = to_fourier(make_list(bound, len(dim)))
    if any(b not in ('dft', 'dct1', 'dct2', 'dst1', 'dst2') for b in bound):
        raise ValueError('Bounds must correspond to a frequency tranform')
    if len(set(bound)) == 1:
        bound = bound[0]
        if bound == 'dft':
            x = ifftn(x, dim=dim)
            if shift:
                x = fftshift(x, dim=dim)
        elif bound == 'dct1':
            x = idctn1(x, dim=dim)
        elif bound == 'dct2':
            x = idctn2(x, dim=dim)
        elif bound == 'dst1':
            x = idstn1(x, dim=dim)
        elif bound == 'dst2':
            x = idstn2(x, dim=dim)
    else:
        for d, b in zip(dim, bound):
            if b == 'dft':
                x = ifftn(x, dim=d)
                if shift:
                    x = fftshift(x, dim=d)
            elif b == 'dct1':
                x = idctn1(x, dim=d)
            elif b == 'dct2':
                x = idctn2(x, dim=d)
            elif b == 'dst1':
                x = idstn1(x, dim=d)
            elif b == 'dst2':
                x = idstn2(x, dim=d)
    return torch.real(x)


def _anyftn_kernel(kernel, shape, dft=True, inv=False):
    """
    Apply any forward frequency tranform to an odd-sized kernel.

    Parameters
    ----------
    kernel : ([ndim, [ndim]], *kershape) tensor
        Input kernel. Kernels can be:

        * "scaled",     when `shape=kernelsize`
        * "diagonal",   when `shape=(ndim, *kernelsize)`
        * "full",       when `shape=(ndim, ndim, *kernelsize)`
    shape : list[int]
        Full shape
    dft : [list of] bool
        If True, appy an FFT. Otherwise, apply a DCT1 to diagonal
        elements and a DST1 to off-diagonal elements.
    inv : bool
        Invert the kernel once it is transformed.

    Returns
    -------
    kernel : ([ndim, [ndim]], *shape) tensor
        Transformed kernel.
    """
    ndim = len(shape)
    dft = make_list(dft, ndim)
    fullshape = [*kernel.shape[:-ndim], *shape]
    kerdim = dict(dim=list(range(-ndim, 0)))

    # We have 2x3 cases that must be dealt with.
    #
    # First, the kernel can be a full matrix of size DxD, a diagonal
    # kernel of size D, or a scalar kernel with no channels.
    # - In the second and third case, all elements are treated the same,
    #   up to their boundary mode (FFT or DCT).
    # - In the first case, however, diagonal and off-diagonal elements
    #   must be treated differently (DST instead of DCT)
    #
    # Then, there are three main ways to deal with boundary modes
    # - All dimensions use an FFT, and we can use a ND-FFT
    # - All dimensions use a DCT, and we can use a ND-DCT
    # - Otherwise, we need to process dimensions one after the other

    # -----------
    # Full matrix
    # -----------
    if kernel.ndim != ndim+2:

        # ------
        # ND-FFT
        # ------
        if all(dft):
            kernel = ensure_shape(kernel, fullshape, ceil=True, side='both')
            kernel = ifftshift(kernel, **kerdim)
            kernel = torch.real(fftn(kernel, **kerdim))

        # ------
        # ND-DCT
        # ------
        elif not any(dft):
            slicer = [slice((k-1//2), None) for k in kernel.shape[-ndim:]]
            kernel = kernel[(Ellipsis, *slicer)]
            kernel = ensure_shape(kernel, fullshape, side='post')
            kernel = dctn1(kernel, **kerdim)

        # -----
        #  MIX
        # -----
        else:
            smallker = kernel
            kernel = kernel.new_zeros(fullshape)
            inp_slicer = [
                slice(None) if f else slice((k-1//2), None)
                for f, k in zip(dft, kernel.shape[-ndim:])
            ]
            out_slicer = [
                slice(int(math.ceil((s-k)/2)), int(math.ceil((s-k)/2)) + k)
                if f else slice((k-1//2))
                for f, k, s in zip(dft, kernel.shape[-ndim:], shape)
            ]
            kernel[(Ellipsis, *out_slicer)] = smallker[(Ellipsis, *inp_slicer)]

        # ---------
        # Inversion
        # ---------
        if inv:
            kernel.clamp_min_(1e-8 * kernel.abs().max()).reciprocal_()

    # -------------------------
    # Diagonal matrix or scalar
    # -------------------------
    else:
        # ------
        # ND-FFT
        # ------
        if all(dft):
            kernel = ensure_shape(kernel, fullshape, ceil=True, side='both')
            kernel = ifftshift(kernel, **kerdim)
            kernel = (fftn(kernel, **kerdim))  # torch.real

        # -------
        # DCT/DST
        # -------
        elif not any(dft):
            smallkernel = kernel
            kernel = kernel.new_zeros(fullshape)
            slicer_iu = [slice((k+1)//2, None) for k in kernel.shape[-ndim:]]
            slicer_id = [slice((k-1)//2, None) for k in kernel.shape[-ndim:]]
            slicer_ou = [slice((k-1)//2) for k in kernel.shape[-ndim:]]
            slicer_od = [slice((k+1)//2) for k in kernel.shape[-ndim:]]
            # off-digonal
            kernel[(Ellipsis, *slicer_ou)] \
                = smallkernel[(Ellipsis, *slicer_iu)]
            # diagonal
            kernel.diagonal(0, -1, -2)[(Ellipsis, *slicer_od)] \
                = smallkernel.diagonal(0, -1, -2)[(Ellipsis, *slicer_id)]
            # DCT1 of diagonal elements
            kernel.diagonal(0, -1, -2).copy_(
                dctn1(kernel.diagonal(0, -1, -2), **kerdim)
            )
            # DST1 of off-diagonal elements
            for d in range(ndim):
                for dd in range(d+1, ndim):
                    kernel[d, dd] = dstn1(kernel[d, dd], **kerdim)
                    kernel[dd, d] = dstn1(kernel[dd, d], **kerdim)

        # -----
        #  MIX
        # -----
        else:
            smallkernel = kernel
            kernel = kernel.new_zeros(fullshape)
            slicer_iu = [slice(None) if f else slice((k+1)//2, None)
                         for f, k in zip(dft, kernel.shape[-ndim:])]
            slicer_id = [slice(None) if f else slice((k-1)//2, None)
                         for f, k in zip(dft, kernel.shape[-ndim:])]
            slicer_ou = [slice(None) if f else slice((k-1)//2)
                         for f, k in zip(dft, kernel.shape[-ndim:])]
            slicer_od = [slice(None) if f else slice((k+1)//2)
                         for f, k in zip(dft, kernel.shape[-ndim:])]

            for d in range(ndim):
                kernel[(d, d, *slicer_od)] = kernel[(d, d, *slicer_id)]
                for dd in range(d+1, ndim):
                    kernel[(d, dd, *slicer_ou)] = kernel[(d, dd, *slicer_iu)]
                    kernel[(dd, d, *slicer_ou)] = kernel[(dd, d, *slicer_iu)]
            dimshift = [2+d for d, f in enumerate(dft) if f]
            kernel = ifftshift(kernel, dim=dimshift)

            for d in range(ndim):
                for i, f in enumerate(dft):
                    fn = ((lambda x: torch.real(fftn(x, dim=2+i))) if f else
                          (lambda x: dctn1(x, dim=2+i)))
                    kernel[d, d] = fn(kernel[d, d])
                for dd in range(d+1, ndim):
                    for i, f in enumerate(dft):
                        fn = ((lambda x: torch.real(fftn(x, dim=2+i))) if f
                              else (lambda x: dstn1(x, dim=2+i)))
                        kernel[d, dd] = fn(kernel[d, dd])
                        kernel[dd, d] = fn(kernel[dd, d])

        # ---------
        # Inversion
        # ---------
        if inv:
            kernel = movedim1(movedim1(kernel, 0, -1), 0, -1)
            kernel = batchinv(kernel)
            kernel = movedim1(movedim1(kernel, -1, 0), -1, 0)

    return kernel
