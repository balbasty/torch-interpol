import torch
from torch.autograd import gradcheck
from interpol import grid_pull, grid_push, grid_count, grid_grad, add_identity_grid_
import pytest

# global parameters
dtype = torch.double        # data type (double advised to check gradients)
shape1 = 3                  # size along each dimension
extrapolate = True

# parameters
bounds = list(range(7))
orders = list(range(8))
devices = [('cpu', 1)]
if torch.backends.openmp.is_available() or torch.backends.mkl.is_available():
    print('parallel backend available')
    devices.append(('cpu', 10))
if torch.cuda.is_available():
    print('cuda backend available')
    devices.append('cuda')
dims = [1, 2, 3]


def make_data(shape, device, dtype):
    grid = torch.randn([*shape, len(shape)], device=device, dtype=dtype)
    grid = add_identity_grid_(grid)
    vol = torch.randn((1,) + shape, device=device, dtype=dtype)
    return vol, grid


def init_device(device):
    if isinstance(device, (list, tuple)):
        device, param = device
    else:
        param = 1 if device == 'cpu' else 0
    if device == 'cuda':
        torch.cuda.set_device(param)
        torch.cuda.init()
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass
        device = '{}:{}'.format(device, param)
    else:
        assert device == 'cpu'
        torch.set_num_threads(param)
    return torch.device(device)


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("bound", bounds)
@pytest.mark.parametrize("interpolation", orders)
def test_gradcheck_grad(device, dim, bound, interpolation):
    print(f'grad_{dim}d({interpolation}, {bound}) on {device}')
    device = init_device(device)
    shape = (shape1,) * dim
    vol, grid = make_data(shape, device, dtype)
    vol.requires_grad = True
    grid.requires_grad = True
    assert gradcheck(grid_grad, (vol, grid, interpolation, bound, extrapolate),
                     rtol=1., raise_exception=True, check_undefined_grad=False)


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("bound", bounds)
@pytest.mark.parametrize("interpolation", orders)
def test_gradcheck_pull(device, dim, bound, interpolation):
    print(f'pull_{dim}d({interpolation}, {bound}) on {device}')
    device = init_device(device)
    shape = (shape1,) * dim
    vol, grid = make_data(shape, device, dtype)
    vol.requires_grad = True
    grid.requires_grad = True
    assert gradcheck(grid_pull, (vol, grid, interpolation, bound, extrapolate),
                     rtol=1., raise_exception=True, check_undefined_grad=False)


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("bound", bounds)
@pytest.mark.parametrize("interpolation", orders)
def test_gradcheck_push(device, dim, bound, interpolation):
    print(f'push_{dim}d({interpolation}, {bound}) on {device}')
    device = init_device(device)
    shape = (shape1,) * dim
    vol, grid = make_data(shape, device, dtype)
    vol.requires_grad = True
    grid.requires_grad = True
    assert gradcheck(grid_push, (vol, grid, shape, interpolation, bound, extrapolate),
                     rtol=1., raise_exception=True, check_undefined_grad=False)


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("bound", bounds)
@pytest.mark.parametrize("interpolation", orders)
def test_gradcheck_count(device, dim, bound, interpolation):
    print(f'count_{dim}d({interpolation}, {bound}) on {device}')
    device = init_device(device)
    shape = (shape1,) * dim
    _, grid = make_data(shape, device, dtype)
    grid.requires_grad = True
    assert gradcheck(grid_count, (grid, shape, interpolation, bound, extrapolate),
                     rtol=1., raise_exception=True, check_undefined_grad=False)