import glob
import os
import torch
import nibabel as nib
from torch.utils.data import Dataset


class PairedDataset(Dataset):
    """A dataset of pairs of images"""

    def __init__(self, fnames, subset=None):
        super().__init__()
        if isinstance(fnames, str):
            if '*' not in fnames:
                fnames = os.path.join(fnames, '*')
            fnames = list(sorted(glob.glob(fnames)))
        fnames = list(fnames)
        if subset:
            fnames = fnames[subset]
        self.fnames = fnames

    def __len__(self):
        return len(self.fnames) ** 2

    def __getitem__(self, index):
        i, j = index // len(self.fnames), index % len(self.fnames)
        fix, mov = self.fnames[i], self.fnames[j]
        fix = torch.as_tensor(
            nib.load(fix).get_fdata(dtype='float32').squeeze()[None]
        )
        mov = torch.as_tensor(
            nib.load(mov).get_fdata(dtype='float32').squeeze()[None]
        )
        fix /= fix.max()
        mov /= mov.max()
        return fix, mov


vxm_folder = '/autofs/cluster/vxmdata1/FS_Slim/proc/cleaned'
# vxm_oasis = list(sorted(glob.glob(
#     f'{vxm_folder}/OASIS_*/norm_talairach_slice.mgz'
# )))
vxm_oasis = list(sorted(glob.glob(
    '/Users/balbasty/Dropbox/Workspace/data/OASIS_2D/*.mgz'
)))

vxm_oasis_train = PairedDataset(vxm_oasis, subset=slice(200))
vxm_oasis_eval = PairedDataset(vxm_oasis, subset=slice(200, 300))
vxm_oasis_test = PairedDataset(vxm_oasis, subset=slice(300, None))
