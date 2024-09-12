import torch
import math
from collections import namedtuple
from bounds import pad, to_fourier
from interpol import (
    resize,
    grid_pull,
    spline_coeff_nd,
    add_identity_grid,
    affine_grid,
    flowreg,
    flowimom,
    flow_upsample2,
    spline_from_coeff_nd,
)
from interpol.energies_nofft import make_evalkernels1d
from interpol.utils import make_list
from lbfgs import LBFGS
from torch.optim import SGD
import matplotlib.pyplot as plt
torch.autograd.set_detect_anomaly(True)

OrientedTensor = namedtuple('OrientedTensor', ['dat', 'affine'])


REDUCE = 'sum'
UP = ''


def resize_affine(affine, shape, new_shape):
    affine = affine.clone()
    scale = torch.as_tensor([s0/s1 for s0, s1 in zip(shape, new_shape)]).to(affine)
    affine[:-1, :-1] *= scale
    affine[:-1, -1] += affine[:-1, :-1].matmul(0.5 * (scale - 1)[:, None])[:, 0]
    return affine


def make_pyramid(
    oriented_tensor,
    *,
    max_levels=None,
    first_level=0,
    min_shape=0,
    bound='dct2',
):
    """
    Generate an image pyramid.

    The edges of the field of view across all levels are aligned.

    Parameters
    ----------
    oriented_tensor : OrientedTensor
        dat : (*spatial, C) tensor
        affine : (D+1, D+1) tensor
    max_levels : int
        Maximum number of levels
    first_level : int
        First level to yield
    min_shape : int
        Stop pyramid when all dimensions are smaller than this number
    bound : [list of] bound-like
        A boundary type from torch-bounds

    Yields
    ------
    OrientedTensor
    """
    dat, affine = oriented_tensor
    dat = dat.movedim(-1, 0)
    shape, ndim = dat.shape[1:], dat.ndim-1
    level = 0

    if first_level == 0:
        yield OrientedTensor(dat, affine)
    level += 1

    kernel = kernel1 = torch.as_tensor([0.25, 0.5, 0.25]).to(dat)
    for _ in range(ndim-1):
        kernel1 = kernel[..., None]
        kernel = kernel * kernel1
    kernel = kernel.expand([len(dat), 1, *kernel.shape])
    conv = getattr(torch.nn.functional, f'conv{ndim}d')
    convopt = dict(padding='same', groups=len(dat))
    paddat = lambda x: x

    bound = to_fourier(make_list(bound, ndim))
    if all(b == 'zero' for b in bound):
        convopt['padding'] = 'same'
    else:
        paddat = lambda x: pad(x, [0] + [1] * ndim, mode=bound, side='both')

    while level < (max_levels or 1):

        level += 1
        prev_shape = shape

        shape = torch.Size([int(math.ceil(s/2)) for s in prev_shape])
        if any(s < (min_shape or 0) for s in shape):
            break

        dat = paddat(dat)
        dat = conv(dat[None], kernel, **convopt)[0]
        dat = resize(dat, shape=shape, bound=bound, anchor='edges')

        affine = resize_affine(affine, prev_shape, shape)

        yield OrientedTensor(dat.movedim(0, -1), affine)


class Likelihood(torch.nn.Module):
    """Base class for likelihoods"""

    def __call__(self, *args, **kwargs):
        return self.nll(*args, **kwargs)

    def nll(self, fix, mov):
        """Negative log-likelihood

        Parameters
        ----------
        fix : (*spatial, C) tensor
        mov : (*spatial, C) tensor

        Returns
        -------
        nll : () tensor
        """
        pass


class GaussianLikelihood(Likelihood):
    """
    A likelihood of the form

        N(fix | alpha * mov + beta, 1/lam)
    """

    def __init__(self, lam=1, alpha=1, beta=0, reduce=REDUCE):
        """
        Parameters
        ----------
        lam : float or ([[*spatial], C]) tensor
            Gaussian precision
        alpha : float or ([[*spatial], C]) tensor
            Scaling factor
        beta : float or ([[*spatial], C]) tensor
            Shift
        reduce : {'mean', 'sum'}
            Whether to compute the average or sum across spatial elements
        """
        super().__init__()
        self.lam = lam
        self.alpha = alpha
        self.beta = beta
        self.reduce = reduce

    def nll(self, fix, mov):
        """Negative log-likelihood"""
        lam = torch.as_tensor(self.lam).to(mov).expand(fix.shape)
        alpha = torch.as_tensor(self.alpha).to(mov)
        beta = torch.as_tensor(self.beta).to(mov)
        mov = alpha * mov + beta
        nll = ((fix-mov).square() * lam).sum()
        nll -= lam.log().sum()
        if self.reduce == 'mean':
            nll /= fix.shape[:-1].numel()
        return 0.5 * nll

    def mstep(self, fix, mov):
        """ML update of the distribution parameters"""
        sdim = list(range(fix.ndim-1))

        f = fix.sum(sdim)
        m = mov.sum(sdim)
        mm = mov.square().sum(sdim)
        self.alpha = (1 - m) * f / (mm - m*m)   #.clamp_min_(1e-8)
        self.beta = (f - self.alpha * m) / fix.shape[:-1].numel()

        mov = self.alpha * mov + self.beta
        self.lam = (fix-mov).var(sdim).clamp_min_(1e-8).reciprocal()


class SplineEnergy:
    """
    Compute the regularization of a spline-encoded displacement field
    """

    def __init__(self, absolute=1e-3, membrane=1, bending=0, div=0, shears=0,
                 reduce=REDUCE, mode='ana'):
        # type: ana, eval, diff
        self.absolute = absolute
        self.membrane = membrane
        self.bending = bending
        self.shears = shears
        self.div = div
        self.reduce = reduce
        self.mode = mode


class SplineDisp(torch.nn.Module):
    """
    A displacement field encoded by B-splines
    """

    def __init__(self, shape, affine, bound='circulant', order=3, energy=SplineEnergy(), up=UP):
        """
        Parameters
        ----------
        shape : list[int]
            Spatial shape of the field
        affine : (D+1, D+1) tensor
            Voxel-to-world matrix of the field
        bound : {'neumann', 'dirichlet', 'circular'}
            Boundary condition
        order : int
            Spline order
        """
        super().__init__()
        ndim = len(shape)
        self.coeff = torch.nn.Parameter(torch.zeros([*shape, ndim]))
        self.affine = affine
        self.bound = bound
        self.order = order
        self.energy = energy
        self.up = up

    def exp(self):
        """
        Return a displacement field, in voxels
        """
        return self.coeff

    def make_full(self, fix_shape, fix_affine, mov_affine):
        """
        Return a transformation field, from fixed voxels to moving voxels

        Parameters
        ----------
        fix_shape : list[int]
            Spatial shape of the fixed image
        fix_affine : (D+1, D+1) tensor
            Voxel-to-world matrix of the fixed image
        mov_affine : (D+1, D+1) tensor
            Voxel-to-world matrix of the moving image

        Returns
        -------
        field : (*fix_shape, D) tensor
            Transformation field

        """
        fix_affine = self.affine.inverse() @ fix_affine.to(self.affine)
        mov_affine = mov_affine.to(self.affine).inverse() @ self.affine

        field = affine_grid(fix_affine, fix_shape)

        field = field + grid_pull(
            self.exp().movedim(-1, 0),
            field,
            interpolation=self.order,
            bound=self.bound,
            extrapolate=True
        ).movedim(0, -1)

        field = mov_affine[:-1, :-1].matmul(field[..., None])[..., 0]
        field += mov_affine[:-1, -1]
        return field

    def make_identity(self, fix_shape, fix_affine, mov_affine):
        """
        Return an idenity transformation field, from fixed to moving voxels

        Parameters
        ----------
        fix_shape : list[int]
            Spatial shape of the fixed image
        fix_affine : (D+1, D+1) tensor
            Voxel-to-world matrix of the fixed image
        mov_affine : (D+1, D+1) tensor
            Voxel-to-world matrix of the moving image

        Returns
        -------
        field : (*fix_shape, D) tensor
            Transformation field

        """
        fix_affine = fix_affine.to(self.affine)
        mov_affine = mov_affine.to(self.affine).inverse()
        field = affine_grid(mov_affine @ fix_affine, fix_shape)
        return field

    def nll(self):
        """
        Prior Negative log-likelihood
        """
        kernels1d = None
        fd = False
        if self.energy.mode == 'eval':
            kernels1d = make_evalkernels1d(self.order, dtype=self.coeff.dtype, device=self.coeff.device)
        elif self.energy.mode == 'diff':
            fd = True
        else:
            assert self.energy.mode == 'ana'
        return flowreg(
            self.coeff,
            order=self.order,
            bound=self.bound,
            absolute=self.energy.absolute,
            membrane=self.energy.membrane,
            bending=self.energy.bending,
            div=self.energy.div,
            shears=self.energy.shears,
            voxel_size=self.affine[:-1, :-1].square().sum(0).sqrt(),
            norm=(self.energy.reduce == 'mean'),
            kernels1d=kernels1d,
            fd=fd,
        )

    def precond(self, grad):
        """
        Hilbert preconditioner
        """
        kernels1d = None
        fd = False
        if self.energy.mode == 'eval':
            kernels1d = make_evalkernels1d(self.order, dtype=self.coeff.dtype, device=self.coeff.device)
        elif self.energy.mode == 'diff':
            fd = True
        else:
            assert self.energy.mode == 'ana'
        alpha = (
            self.energy.absolute +
            self.energy.membrane +
            self.energy.bending +
            self.energy.div +
            self.energy.shears
        )
        return flowimom(
            grad,
            order=self.order,
            bound=self.bound,
            absolute=self.energy.absolute,
            membrane=self.energy.membrane,
            bending=self.energy.bending,
            div=self.energy.div,
            shears=self.energy.shears,
            voxel_size=self.affine[:-1, :-1].square().sum(0).sqrt(),
            norm=(self.energy.reduce == 'mean'),
            kernels1d=kernels1d,
            fd=fd
        ) # / alpha

    def up2(self):
        """
        Upsample the field by a factor two, while minimizing the MSE
        """
        ndim = self.coeff.shape[-1]
        prev_shape = self.coeff.shape[-ndim-1:-1]
        if self.up == 'mse':
            self.coeff = torch.nn.Parameter(flow_upsample2(
                self.coeff,
                order=self.order,
                bound=self.bound,
            ))
        else:
            self.coeff = torch.nn.Parameter(
                spline_coeff_nd(
                    resize(
                        self.coeff.movedim(-1, 0),
                        shape=[2*s for s in prev_shape],
                        anchor='edges',
                        interpolation=self.order,
                        bound=self.bound,
                        prefilter=False,
                    ),
                    interpolation=self.order,
                    bound=self.bound,
                    dim=ndim,
                    inplace=True,
                ).movedim(0, -1).mul_(2).to(
                    memory_format=torch.contiguous_format, copy=True
                )
            )

        # update affine
        affine = self.affine.clone()
        affine[:-1, :-1] *= 0.5
        affine[:-1, -1] += affine[:-1, :-1].sum(-1).mul_(-0.25)
        self.affine = affine
        return self


class SplineSVF(SplineDisp):
    """
    A stationary velocity field encoded by B-splines
    """

    def __init__(self, shape, affine, steps=8, bound='circulant', order=3,
                 energy=SplineEnergy(), up=UP):
        """
        Parameters
        ----------
        shape : list[int]
            Spatial shape of the field
        affine : (D+1, D+1) tensor
            Voxel-to-world matrix of the field
        steps : int
            number of scaling and squaring steps
        bound : {'neumann', 'dirichlet', 'circular'}
            Boundary condition
        order : int
            Spline order
        """
        super().__init__(shape, affine, bound=bound, order=order, energy=energy, up=up)
        self.steps = steps

    def exp(self):
        """
        Return a displacement field, in voxels
        """
        field = self.coeff
        for _ in range(self.steps-1):
            field = field + spline_coeff_nd(
                grid_pull(field.movedim(-1, 0), add_identity_grid(field),
                          interpolation=self.order, bound=self.bound,
                          extrapolate=False).movedim(0, -1),
                interpolation=self.order, bound=self.bound,
                dim=field.shape[-1],
                inplace=True,
            )
        return field


class RegistrationEngine:
    """
    The class that does the actual registration
    """

    def __init__(
        self,
        likelihood=GaussianLikelihood(),
        prior=SplineEnergy(),
        max_iter=64,
        tol=1e-12,
        min_level_image=0,
        max_level_image=4,
        min_level_flow=0,
        max_level_flow=4,
        exp_steps=0,
        optim='lbfgs',
        precond=True,
        up=UP,
    ):
        """
        Parameters
        ----------
        likelihood : Likelihood
            Data term
        prior : Likelihood
            Regularization term
        max_iter : int
            Maximum number of iterations per level
        tol : float
            Tolerance for early stopping
        min_level_image : int
            Minimum pyramid level of the images
        max_level_image : int
            Maximum pyramid level of the images
        min_level_flow : int
            Minimum pyramid level of the flow
        max_level_flow : int
            Maximum pyramid level of the flow
        exp_steps : int
            Number of scaling and squaring steps. If 0: simple displacement.
        """
        self.likelihood = likelihood
        self.prior = prior
        self.max_iter = max_iter
        self.tol = tol
        self.min_level_image = min_level_image
        self.max_level_image = max_level_image
        self.min_level_flow = min_level_flow
        self.max_level_flow = max_level_flow
        self.exp_steps = exp_steps
        self.optim = optim
        self.precond = precond
        self.up = up

    def warp(self):
        flow = self.flow.make_full(
            self.fix.dat.shape[:-1],
            self.fix.affine,
            self.mov.affine
        )
        moved = grid_pull(
            self.mov.dat.movedim(-1, 0),
            flow,
            interpolation=1,
            bound='dct2',
            extrapolate=False
        ).movedim(0, -1)
        return moved, flow

    def warp_identity(self):
        flow = self.flow.make_identity(
            self.fix.dat.shape[:-1],
            self.fix.affine,
            self.mov.affine
        )
        moved = grid_pull(
            self.mov.dat.movedim(-1, 0),
            flow,
            interpolation=1,
            bound='dct2',
            extrapolate=False
        ).movedim(0, -1)
        return moved, flow

    def closure(self, backward=True, save=False):
        """Compute the negative log-likelihood with the current flow"""
        self.flow.coeff.requires_grad_()
        self.flow.coeff.grad = None
        moved, _ = self.warp()
        self.likelihood.mstep(self.fix.dat, moved)
        nll = self.likelihood(self.fix.dat, moved)
        nlp = self.flow.nll()
        if save:
            self._all_nll.append(float(nll))
            self._all_nlp.append(float(nlp))
            self._all_nl.append(float(nll) + float(nlp))
        print('search:', nll.item(), '+', nlp.item(), end=' = ')
        nll += nlp
        print(nll.item())
        if backward:
            nll.backward()
        return nll

    def iter(self):
        nlls = []
        nll0 = float('inf')
        for _ in range(self.max_iter):
            nll = self.closure(save=True)
            nlls.append(float(nll))
            if ((nll-nll0)/nll0).abs() < self.tol:
                break
            if nll < nll0:
                self.optim.param_groups[0]['lr'] *= 1.5
            else:
                self.optim.param_groups[0]['lr'] *= 0.1
            nll0 = nll
            self._optim.step(self.closure)
        with torch.no_grad():
            nlls.append(float(self.closure(backward=False, save=True)))
        return nlls

    def iter_pyramid(self):
        """Perform optimization across all levels"""
        self.all_losses = []
        self.all_likelihoods = []
        self.all_priors = []
        min_level = min(self.min_level_flow, self.min_level_image)
        max_level_flow = min(self.max_level_flow, min_level + len(self.allfix)-1)
        for i, (self.fix, self.mov) in enumerate(zip(self.allfix, self.allmov)):
            level = min_level
            level += len(self.allfix) - i - 1
            print(f'--- level {level} ---')
            if self.min_level_flow <= level < max_level_flow:
                with torch.no_grad():
                    print('before up')
                    self.closure(backward=False)
                    self.flow.up2()
                    print('after up')
                    self.closure(backward=False)
            print(self.flow.coeff.shape)

            self._all_nll = []
            self._all_nlp = []
            self._all_nl = []

            if self.optim == 'lbfgs':
                self._optim = LBFGS(
                    self.flow.parameters(),
                    max_iter=self.max_iter,
                    tolerance_change=self.tol,
                    tolerance_grad=0,
                    history_size=16,
                    line_search_fn='strong_wolfe',
                    precond=self.flow.precond if self.precond else None,
                )
                self._optim.step(self.closure)

            elif self.optim == 'gd':
                self.optim = SGD(
                    self.flow.parameters(),
                    lr=10,
                )
                # self.all_losses.append(self.iter())

            self.all_losses.append(list(self._all_nl))
            self.all_likelihoods.append(list(self._all_nll))
            self.all_priors.append(list(self._all_nlp))

            self.plot_loss(*self.all_losses)
            self.plot()

    def plot_loss(self, *losses):
        n = 0
        for loss in losses:
            plt.plot(list(range(n, n+len(loss))), loss)
            n += len(loss)
        plt.legend(list(map(str, range(len(losses)))))
        plt.show()

    def plot(self):
        with torch.no_grad():
            moved, flow = self.warp()
            moving, idt = self.warp_identity()
            flow -= idt
            if moved.ndim == 4:
                z = moved.shape[-2] // 2
                moved = moved[..., z, 0]
                moving = moving[..., z, 0]
                fixed = self.fix.dat[..., z, 0]
                flow = flow[..., z, :].square().sum(-1).sqrt()
            else:
                moved = moved[..., 0]
                fixed = fixed[..., 0]
                moving = moving[..., 0]
                flow = flow.square().sum(-1).sqrt()
        plt.subplot(1, 4, 1)
        plt.imshow(moving)
        plt.axis('off')
        plt.subplot(1, 4, 2)
        plt.imshow(moved)
        plt.axis('off')
        plt.subplot(1, 4, 3)
        plt.imshow(fixed)
        plt.axis('off')
        plt.subplot(1, 4, 4)
        plt.imshow(flow)
        plt.axis('off')
        plt.colorbar()
        plt.tight_layout()
        plt.show()
        foo = 0

    def init(self, fix, mov):
        """
        Initialize state before registering mov to fix.

        - Computes the image pyramids if needed
        - Initializes the displacement field

        Parameters
        ----------
        fix : OrientedTensor
            dat : (*spatial, C) tensor
            affine : (D+1, D+1) tensor
        mov : OrientedTensor
            dat : (*spatial, C) tensor
            affine : (D+1, D+1) tensor
        """
        MIN_SHAPE = 7  # kernel size for cubic spline

        max_levels = 1 + max(self.max_level_image, self.max_level_flow)
        allfix = list(make_pyramid(fix, max_levels=max_levels, min_shape=MIN_SHAPE))

        # initial flow
        max_level_flow = min(self.max_level_flow, len(allfix)-1)
        max_level_image = min(self.max_level_image, len(allfix))
        shape = allfix[max_level_flow].dat.shape[:-1]
        affine = allfix[max_level_flow].affine
        if self.exp_steps:
            flow = SplineSVF(shape, affine, self.exp_steps, energy=self.prior, up=self.up)
        else:
            flow = SplineDisp(shape, affine, energy=self.prior, up=self.up)

        # image pyramid
        allmov = list(make_pyramid(mov, max_levels=1+self.max_level_image, min_shape=MIN_SHAPE))
        allfix = allfix[self.min_level_image:1+self.max_level_image]
        allmov = allmov[self.min_level_image:1+self.max_level_image]
        if self.min_level_image > self.min_level_flow:
            allfix = allfix[:1] * (self.min_level_image - self.min_level_flow) + allfix
            allmov = allmov[:1] * (self.min_level_image - self.min_level_flow) + allmov
        if max_level_image < max_level_flow:
            allfix += allfix[-1:] * (max_level_flow - max_level_image)
            allmov += allmov[-1:] * (max_level_flow - max_level_image)

        self.allfix = allfix[::-1]
        self.allmov = allmov[::-1]
        self.flow = flow

    def __call__(self, fix, mov):
        self.init(fix, mov)
        self.iter_pyramid()
        return self.flow
