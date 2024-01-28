"""AutoGrad version of pull/push/count/grad"""
import torch
from .coeff import spline_coeff_nd, spline_coeff
from .bounds import to_int as bound_to_int
from .splines import to_int as inter_to_int
from .pushpull import (
    grid_pull, grid_pull_backward,
    grid_push, grid_push_backward,
    grid_count, grid_count_backward,
    grid_grad, grid_grad_backward)
try:
    from torch.cuda.amp import custom_fwd, custom_bwd
except (ModuleNotFoundError, ImportError):
    from .utils import fake_decorator
    custom_fwd = custom_bwd = fake_decorator


def make_list(x):
    if not isinstance(x, (list, tuple)):
        x = [x]
    return list(x)


class GridPull(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, grid, interpolation, bound, extrapolate):

        bound = bound_to_int(make_list(bound))
        interpolation = inter_to_int(make_list(interpolation))
        extrapolate = int(extrapolate)
        opt = (bound, interpolation, extrapolate)

        # Pull
        output = grid_pull(input, grid, *opt)

        # Context
        ctx.opt = opt
        ctx.save_for_backward(input, grid)

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        var = ctx.saved_tensors
        opt = ctx.opt
        grads = grid_pull_backward(grad, *var, *opt)
        grad_input, grad_grid = grads
        return grad_input, grad_grid, None, None, None


class GridPush(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, grid, shape, interpolation, bound, extrapolate):

        bound = bound_to_int(make_list(bound))
        interpolation = inter_to_int(make_list(interpolation))
        extrapolate = int(extrapolate)
        opt = (bound, interpolation, extrapolate)

        # Push
        output = grid_push(input, grid, shape, *opt)

        # Context
        ctx.opt = opt
        ctx.save_for_backward(input, grid)

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        var = ctx.saved_tensors
        opt = ctx.opt
        grads = grid_push_backward(grad, *var, *opt)
        grad_input, grad_grid = grads
        return grad_input, grad_grid, None, None, None, None


class GridCount(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, grid, shape, interpolation, bound, extrapolate):

        bound = bound_to_int(make_list(bound))
        interpolation = inter_to_int(make_list(interpolation))
        extrapolate = int(extrapolate)
        opt = (bound, interpolation, extrapolate)

        # Push
        output = grid_count(grid, shape, *opt)

        # Context
        ctx.opt = opt
        ctx.save_for_backward(grid)

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        var = ctx.saved_tensors
        opt = ctx.opt
        grad_grid = None
        if ctx.needs_input_grad[0]:
            grad_grid = grid_count_backward(grad, *var, *opt)
        return grad_grid, None, None, None, None


class GridGrad(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, grid, interpolation, bound, extrapolate):

        bound = bound_to_int(make_list(bound))
        interpolation = inter_to_int(make_list(interpolation))
        extrapolate = int(extrapolate)
        opt = (bound, interpolation, extrapolate)

        # Pull
        output = grid_grad(input, grid, *opt)

        # Context
        ctx.opt = opt
        ctx.save_for_backward(input, grid)

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        var = ctx.saved_tensors
        opt = ctx.opt
        grad_input = grad_grid = None
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            grads = grid_grad_backward(grad, *var, *opt)
            grad_input, grad_grid = grads
        return grad_input, grad_grid, None, None, None


class SplineCoeff(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, input, bound, interpolation, dim, inplace):

        bound = bound_to_int(make_list(bound)[0])
        interpolation = inter_to_int(make_list(interpolation)[0])
        opt = (bound, interpolation, dim, inplace)

        # Pull
        output = spline_coeff(input, *opt)

        # Context
        if input.requires_grad:
            ctx.opt = opt

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        # symmetric filter -> backward == forward
        # (I don't know if I can write into grad, so inplace=False to be safe)
        grad = spline_coeff(grad, *ctx.opt[:-1], inplace=False)
        return [grad] + [None] * 4


class SplineCoeffND(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, input, bound, interpolation, dim, inplace):

        bound = bound_to_int(make_list(bound))
        interpolation = inter_to_int(make_list(interpolation))
        opt = (bound, interpolation, dim, inplace)

        # Pull
        output = spline_coeff_nd(input, *opt)

        # Context
        if input.requires_grad:
            ctx.opt = opt

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        # symmetric filter -> backward == forward
        # (I don't know if I can write into grad, so inplace=False to be safe)
        grad = spline_coeff_nd(grad, *ctx.opt[:-1], inplace=False)
        return grad, None, None, None, None
