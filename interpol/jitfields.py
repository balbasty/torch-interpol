try:
    import jitfields
    available = True
    from jitfields.pushpull import (
        pull as jitpull, 
        push as jitpush, 
        count as jitcount, 
        grad as jitgrad,
    )
    from jitfields.resize import (
        resize as jitresize, 
        restrict as jitrestrict,
    )
    from jitfields.splinc import (
        spline_coeff as jitcoeff,
        spline_coeff_nd as jitcoeffnd,
        spline_coeff_ as jitcoeff_,
        spline_coeff_nd_ as jitcoeffnd_,
    )
except (ImportError, ModuleNotFoundError):
    jitfields = None
    available = False
    jitpull = jitpush = jitcount = jitgrad = None
    jitcoeff = jitcoeff_ = jitcoeffnd = jitcoeffnd_ = None
    jitresize = jitrestrict = None
from .utils import make_list
import torch


def first2last(input, ndim):
    insert = input.dim() <= ndim
    if insert:
        input = input.unsqueeze(-1)
    else:
        input = torch.movedim(input, -ndim-1, -1)
    return input, insert


def last2first(input, ndim, inserted, grad=False):
    if inserted:
        input = input.squeeze(-1 - grad)
    else:
        input = torch.movedim(input, -1 - grad, -ndim-1 - grad)
    return input


def grid_pull(input, grid, interpolation='linear', bound='zero',
              extrapolate=False, prefilter=False):
    ndim = grid.shape[-1]
    input, inserted = first2last(input, ndim)
    input = jitpull(input, grid, order=interpolation, bound=bound,
                    extrapolate=extrapolate, prefilter=prefilter)
    input = last2first(input, ndim, inserted)
    return input


def grid_push(input, grid, shape=None, interpolation='linear', bound='zero',
              extrapolate=False, prefilter=False):
    ndim = grid.shape[-1]
    input, inserted = first2last(input, ndim)
    input = jitpush(input, grid, shape, order=interpolation, bound=bound,
                    extrapolate=extrapolate, prefilter=prefilter)
    input = last2first(input, ndim, inserted)
    return input


def grid_count(grid, shape=None, interpolation='linear', bound='zero',
               extrapolate=False):
    return jitcount(grid, shape, order=interpolation, bound=bound,
                    extrapolate=extrapolate)


def grid_grad(input, grid, interpolation='linear', bound='zero',
              extrapolate=False, prefilter=False):
    ndim = grid.shape[-1]
    input, inserted = first2last(input, ndim)
    input = jitgrad(input, grid, order=interpolation, bound=bound,
                    extrapolate=extrapolate, prefilter=prefilter)
    input = last2first(input, ndim, inserted, True)
    return input


def spline_coeff(input, interpolation='linear', bound='dct2', dim=-1,
                 inplace=False):
    func = jitcoeff_ if inplace else jitcoeff
    return func(input, interpolation, bound=bound, dim=dim)


def spline_coeff_nd(input, interpolation='linear', bound='dct2', dim=None,
                    inplace=False):
    func = jitcoeffnd_ if inplace else jitcoeffnd
    return func(input, interpolation, bound=bound, ndim=dim)


def resize(image, factor=None, shape=None, anchor='c',
           interpolation=1, prefilter=True, **kwargs):
    kwargs.setdefault('bound', 'nearest')
    ndim = max(len(make_list(factor or [])),
               len(make_list(shape or [])),
               len(make_list(anchor or []))) or (image.dim() - 2)
    return jitresize(image, factor=factor, shape=shape, ndim=ndim,
                     anchor=anchor, order=interpolation,
                     bound=kwargs['bound'], prefilter=prefilter)


def restrict(image, factor=None, shape=None, anchor='c',
             interpolation=1, reduce_sum=False, **kwargs):
    kwargs.setdefault('bound', 'nearest')
    ndim = max(len(make_list(factor or [])),
               len(make_list(shape or [])),
               len(make_list(anchor or []))) or (image.dim() - 2)
    return jitrestrict(image, factor=factor, shape=shape, ndim=ndim,
                       anchor=anchor, order=interpolation,
                       bound=kwargs['bound'], reduce_sum=reduce_sum)
