import sys
import torch
import nibabel as nib
import matplotlib.pyplot as plt
from register import RegistrationEngine, SplineEnergy, OrientedTensor, GaussianLikelihood

if len(sys.argv) > 1:
    fix, mov, *lam = sys.argv[1:]
    if lam:
        lam = float(lam[0])
    else:
        lam = 0.1
else:
    fix = "/Users/balbasty/localdata/ext/Learn2Reg/OASIS/imagesTr/OASIS_0395_0000.nii.gz"
    mov = "/Users/balbasty/localdata/ext/Learn2Reg/OASIS/imagesTr/OASIS_0396_0000.nii.gz"
    lam = 0.1

fix, mov = nib.load(fix), nib.load(mov)

fix = OrientedTensor(
    torch.as_tensor(fix.get_fdata(), dtype=torch.float32),
    torch.as_tensor(fix.affine)
)
mov = OrientedTensor(
    torch.as_tensor(mov.get_fdata(), dtype=torch.float32),
    torch.as_tensor(mov.affine)
)
if fix.dat.ndim == 3:
    fix.dat.unsqueeze_(-1)
if mov.dat.ndim == 3:
    mov.dat.unsqueeze_(-1)


for precond in (False,):  # True, False:
    for up in ('mse', ''):

        all_losses = {}
        all_likelihoods = {}
        all_priors = {}
        for mode in ('ana', 'eval', 'diff'):
            likelihood = GaussianLikelihood(lam=100)
            prior = SplineEnergy(absolute=lam*1e-5, bending=lam, mode=mode)
            engine = RegistrationEngine(
                likelihood=likelihood,
                prior=prior,
                up=up,
                precond=precond,
                min_level_image=1,
                max_level_image=1,
                min_level_flow=2,
                max_level_flow=6
            )
            engine(fix, mov)
            all_losses[mode] = engine.all_losses
            all_likelihoods[mode] = engine.all_likelihoods
            all_priors[mode] = engine.all_priors

        mnloss = min(min(min(loss for loss in losses) for losses in loss_mode) for loss_mode in all_losses.values())
        mxloss = max(max(max(loss for loss in losses) for losses in loss_mode) for loss_mode in all_losses.values())
        mnll = min(min(min(loss for loss in losses) for losses in loss_mode) for loss_mode in all_likelihoods.values())
        mxll = max(max(max(loss for loss in losses) for losses in loss_mode) for loss_mode in all_likelihoods.values())
        mnlp = min(min(min(loss for loss in losses) for losses in loss_mode) for loss_mode in all_priors.values())
        mxlp = max(max(max(loss for loss in losses) for losses in loss_mode) for loss_mode in all_priors.values())
        plt.figure()
        for i, mode in enumerate(all_losses.keys()):
            plt.subplot(3, 3, i+1)
            n = 0
            for loss in all_priors[mode]:
                plt.plot(list(range(n, n+len(loss))), loss)
                n += len(loss)
            plt.title(mode)
            plt.ylim([mnlp, mxlp])
            plt.subplot(3, 3, i+4)
            n = 0
            for loss in all_likelihoods[mode]:
                plt.plot(list(range(n, n+len(loss))), loss)
                n += len(loss)
            plt.title(mode)
            plt.ylim([mnll, mxll])
            plt.subplot(3, 3, i+7)
            n = 0
            for loss in all_losses[mode]:
                plt.plot(list(range(n, n+len(loss))), loss)
                n += len(loss)
            plt.title(mode)
            plt.ylim([mnloss, mxloss])
        plt.tight_layout()
        plt.suptitle(('[MSE Up] ' if up == 'mse' else '') + ('[Precond]' if precond else ''))
        plt.show(block=False)
        foo = 0
