import torch


def make_list(x, n=None, **kwargs):
    """Ensure that the input  is a list (of a given size)

    Parameters
    ----------
    x : list or tuple or scalar
        Input object
    n : int, optional
        Required length
    default : scalar, optional
        Value to right-pad with. Use last value of the input by default.

    Returns
    -------
    x : list
    """
    if not isinstance(x, (list, tuple)):
        x = [x]
    x = list(x)
    default = kwargs.get('default', x[-1])
    if n:
        x = x + [default] * max(0, n - len(x))
    return x


def expanded_shape(*shapes, side='left'):
    """Expand input shapes according to broadcasting rules

    Parameters
    ----------
    *shapes : sequence[int]
        Input shapes
    side : {'left', 'right'}, default='left'
        Side to add singleton dimensions.

    Returns
    -------
    shape : tuple[int]
        Output shape

    Raises
    ------
    ValueError
        If shapes are not compatible for broadcast.

    """
    def error(s0, s1):
        raise ValueError('Incompatible shapes for broadcasting: {} and {}.'
                         .format(s0, s1))

    # 1. nb dimensions
    nb_dim = 0
    for shape in shapes:
        nb_dim = max(nb_dim, len(shape))

    # 2. enumerate
    shape = [1] * nb_dim
    for i, shape1 in enumerate(shapes):
        pad_size = nb_dim - len(shape1)
        ones = [1] * pad_size
        if side == 'left':
            shape1 = [*ones, *shape1]
        else:
            shape1 = [*shape1, *ones]
        shape = [max(s0, s1) if s0 == 1 or s1 == 1 or s0 == s1
                 else error(s0, s1) for s0, s1 in zip(shape, shape1)]

    return tuple(shape)


def matvec(mat, vec, out=None):
    """Matrix-vector product (supports broadcasting)

    Parameters
    ----------
    mat : (..., M, N) tensor
        Input matrix.
    vec : (..., N) tensor
        Input vector.
    out : (..., M) tensor, optional
        Placeholder for the output tensor.

    Returns
    -------
    mv : (..., M) tensor
        Matrix vector product of the inputs

    """
    vec = vec[..., None]
    if out is not None:
        out = out[..., None]

    mv = torch.matmul(mat, vec, out=out)
    mv = mv[..., 0]
    if out is not None:
        out = out[..., 0]

    return mv