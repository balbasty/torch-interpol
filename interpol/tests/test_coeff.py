from interpol import resize
import pytest
import torch

lengths = [1, 2, 3, 7, 9, 11]
orders = list(range(8))
bounds = ['dct1', 'dct2', 'dft']


@pytest.mark.parametrize("length", lengths)
@pytest.mark.parametrize("bound", bounds)
@pytest.mark.parametrize("order", orders)
def test_identity(length, bound, order):
    x = torch.randn([length], dtype=torch.double)
    y = resize(x, shape=[length], bound=bound, interpolation=order)
    assert torch.allclose(x, y)
