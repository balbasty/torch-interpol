import torch


def fake_decorator(*a, **k):
    if len(a) == 1 and not k:
        return a[0]
    else:
        return fake_decorator


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
    if n and len(x) < n:
        default = kwargs.get('default', x[-1])
        x = x + [default] * max(0, n - len(x))
    return x


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


def prod(sequence, inplace=False):
    """Perform the product of a sequence of elements.

    Parameters
    ----------
    sequence : any object that implements `__iter__`
        Sequence of elements for which the `__mul__` operator is defined.
    inplace : bool, default=False
        Perform the product inplace (using `__imul__` instead of `__mul__`).

    Returns
    -------
    product :
        Product of the elements in the sequence.

    """
    accumulate = None
    for elem in sequence:
        if accumulate is None:
            accumulate = elem
        elif inplace:
            accumulate *= elem
        else:
            accumulate = accumulate * elem
    return accumulate


def _compare_versions(version1, mode, version2):
    for v1, v2 in zip(version1, version2):
        if mode in ('gt', '>'):
            if v1 > v2:
                return True
            elif v1 < v2:
                return False
        elif mode in ('ge', '>='):
            if v1 > v2:
                return True
            elif v1 < v2:
                return False
        elif mode in ('lt', '<'):
            if v1 < v2:
                return True
            elif v1 > v2:
                return False
        elif mode in ('le', '<='):
            if v1 < v2:
                return True
            elif v1 > v2:
                return False
    if mode in ('gt', 'lt', '>', '<'):
        return False
    else:
        return True


def torch_version(mode, version):
    """Check torch version

    Parameters
    ----------
    mode : {'<', '<=', '>', '>='}
    version : tuple[int]

    Returns
    -------
    True if "torch.version <mode> version"

    """
    current_version, *cuda_variant = torch.__version__.split('+')
    major, minor, patch, *_ = current_version.split('.')
    # strip alpha tags
    for x in 'abcdefghijklmnopqrstuvwxy':
        if x in patch:
            patch = patch[:patch.index(x)]
    current_version = (int(major), int(minor), int(patch))
    version = make_list(version)
    return _compare_versions(current_version, mode, version)


@torch.jit.script
def det2(a):
    dt = a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]
    return dt


@torch.jit.script
def det3(a):
    dt = a[0, 0] * (a[1, 1] * a[2, 2] - a[1, 2] * a[2, 1]) + \
         a[0, 1] * (a[1, 2] * a[2, 0] - a[1, 0] * a[2, 2]) + \
         a[0, 2] * (a[1, 0] * a[2, 1] - a[1, 1] * a[2, 0])
    return dt


def batchdet(a):
    """Efficient batched determinant for large batches of small matrices

    !!! note
        A batched implementation is used for 1x1, 2x2 and 3x3 matrices.
        Other sizes fall back to `torch.det`.

    Parameters
    ----------
    a : (..., n, n) tensor
        Input matrix.

    Returns
    -------
    d : (...) tensor
        Determinant.

    """
    if not a.is_cuda or a.shape[-1] > 3:
        return a.det()
    a = a.movedim(-1, 0).movedim(-1, 0)
    if len(a) == 3:
        a = det3(a)
    elif len(a) == 2:
        a = det2(a)
    else:
        assert len(a) == 1
        a = a.clone()[0, 0]
    return a


@torch.jit.script
def inv2(A):
    F = torch.empty_like(A)
    F[0, 0] = A[1, 1]
    F[1, 1] = A[0, 0]
    F[0, 1] = -A[0, 1]
    F[1, 0] = -A[1, 0]
    dt = det2(A)
    Aabs = A.reshape((-1,) + A.shape[2:]).abs()
    rnge = Aabs.max(dim=0).values - Aabs.min(dim=0).values
    dt += rnge * 1E-12
    F /= dt[None, None]
    return F


@torch.jit.script
def inv3(A):
    F = torch.empty_like(A)
    F[0, 0] = A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]
    F[1, 1] = A[0, 0] * A[2, 2] - A[0, 2] * A[2, 0]
    F[2, 2] = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    F[0, 1] = A[0, 2] * A[2, 1] - A[0, 1] * A[2, 2]
    F[0, 2] = A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1]
    F[1, 0] = A[1, 2] * A[2, 0] - A[1, 0] * A[2, 2]
    F[1, 2] = A[1, 0] * A[0, 2] - A[1, 2] * A[0, 0]
    F[2, 0] = A[2, 1] * A[1, 0] - A[2, 0] * A[1, 1]
    F[2, 1] = A[2, 0] * A[0, 1] - A[2, 1] * A[0, 0]
    dt = det3(A)
    Aabs = A.reshape((-1,) + A.shape[2:]).abs()
    rnge = Aabs.max(dim=0).values - Aabs.min(dim=0).values
    dt += rnge * 1E-12
    F /= dt[None, None]
    return F


def batchinv(a):
    """Efficient batched inversion for large batches of small matrices

    !!! note
        A batched implementation is used for 1x1, 2x2 and 3x3 matrices.
        Other sizes fall back to `torch.linagl.inv`.

    Parameters
    ----------
    a : (..., n, n) tensor
        Input matrix.

    Returns
    -------
    a : (..., n, n) tensor
        Inverse matrix.

    """
    if not a.is_cuda or a.shape[-1] > 3:
        return a.inverse()
    a = a.movedim(-1, 0).movedim(-1, 0)
    if len(a) == 3:
        a = inv3(a)
    elif len(a) == 2:
        a = inv2(a)
    else:
        assert len(a) == 1
        a = a.reciprocal()
    a = a.movedim(0, -1).movedim(0, -1)
    return a
