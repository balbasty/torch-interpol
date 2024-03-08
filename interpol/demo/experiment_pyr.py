import torch
import os
import sys
from torch.utils.data import DataLoader
from .models import PyramidMorph
from .datasets import vxm_oasis_train, vxm_oasis_eval, vxm_oasis_test
from .loadable import LoadableModule, saveargs
from .losses import NCC
from ..layers import CoeffToValue, FlowPull, FlowLoss


class PyrTrainer(LoadableModule):

    @saveargs
    def __init__(
            self,
            odir='.',
            order=3,
            nb_epochs=5,  # 200*200 trainig pairs -> 200,000 steps
            lr=1e-4,
            save_every=1,
            lam=0.1,
            batch_size=4,
            device=None,
    ):
        super().__init__()
        self.odir = odir
        self.order = order
        self.nb_epochs = nb_epochs
        self.lr = lr
        self.save_every = save_every
        self.lam = lam
        self.batch_size = batch_size
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        self.model = PyramidMorph(2, self.order)
        self.model.to(self.device)
        self.tovalue = CoeffToValue(interpolation=order, bound='dft')
        self.pull = FlowPull()
        self.loss = NCC()
        self.energy = FlowLoss(membrane=lam, interpolation=order, bound='dft')

        self.trainset = DataLoader(
            vxm_oasis_train, batch_size=batch_size, shuffle=True)
        self.evalset = DataLoader(
            vxm_oasis_eval, batch_size=16, shuffle=False)
        self.testset = DataLoader(
            vxm_oasis_test, batch_size=16, shuffle=False)

        self.optim = torch.optim.Adam(self.model.parameters(), lr)

        # initial state
        self.epoch = 0
        self.best_loss = float('inf')
        self.train_loss = []
        self.eval_loss = []

    def save_checkpoint(self, fname=None, **more_stuff):
        more_stuff['optim'] = self.optim.state_dict()
        more_stuff['epoch'] = self.epoch
        more_stuff['best_loss'] = self.best_loss
        more_stuff['train_loss'] = self.train_loss
        more_stuff['eval_loss'] = self.eval_loss
        return super().save_checkpoint(fname, **more_stuff)

    def load_more_stuff(self, **more_stuff):
        self.optim.load_state_dict(more_stuff['optim'])
        self.epoch = more_stuff['epoch']
        self.best_loss = more_stuff['best_loss']
        self.train_loss = more_stuff['train_loss']
        self.eval_loss = more_stuff['eval_loss']

    def train_epoch(self):
        import matplotlib.pyplot as plt

        self.model.train()
        avg_loss = 0
        for batch, (fix, mov) in enumerate(self.trainset):
            fix, mov = fix.to(self.device), mov.to(self.device)
            flow = self.model(fix, mov)
            reg = self.energy(flow)
            flow = self.tovalue(flow)
            moved = self.pull(mov, flow)
            sim = self.loss(moved, fix)
            loss = sim + self.lam * reg

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            loss, sim, reg = loss.item(), sim.item(), reg.item()
            avg_loss = (batch * avg_loss + loss) / (batch + 1)

            if batch % 40 == 0 or batch == len(self.trainset) - 1:
                print(
                    f'{self.epoch:02d} | train | {batch:05d} | '
                    f'{sim:8.3g} + {self.lam:g} * {reg:8.3g} = {loss:8.3g}'
                    f' (epoch average: {avg_loss:8.3g})', end='\r'
                )
                plt.clf()
                plt.gcf()
                plt.subplot(2, 2, 1)
                plt.imshow(fix[0, 0].detach().cpu())
                plt.axis('off')
                plt.subplot(2, 2, 2)
                plt.imshow(mov[0, 0].detach().cpu())
                plt.axis('off')
                plt.subplot(2, 2, 3)
                plt.imshow(flow[0].detach().square().sum(0).sqrt().cpu())
                plt.axis('off')
                plt.subplot(2, 2, 4)
                plt.imshow(moved[0, 0].detach().cpu())
                plt.axis('off')
                plt.show(block=True)
        print('')
        return avg_loss

    def eval_epoch(self):
        self.model.eval()
        with torch.no_grad():
            avg_loss = 0
            for batch, (fix, mov) in enumerate(self.evalset):
                fix, mov = fix.to(self.device), mov.to(self.device)
                flow = self.model(fix, mov)
                reg = self.energy(flow)
                flow = self.tovalue(flow)
                moved = self.pull(mov, flow)
                sim = self.loss(moved, fix)
                loss = sim + self.lam * reg

                loss, sim, reg = loss.item(), sim.item(), reg.item()
                avg_loss = (batch * avg_loss + loss) / (batch + 1)

                if batch % 40 == 0 or batch == len(self.evalset) - 1:
                    print(f'{self.epoch:02d} | eval  | {batch:05d} | '
                          f'{sim:8.3g} + {self.lam:g} * {reg:8.3g} '
                          f'= {loss:8.3g} '
                          f'(epoch average: {avg_loss:8.3g})', end='\r')
        print('')
        return avg_loss

    def train(self):
        import matplotlib.pyplot as plt
        plt.figure()

        # self.best_loss = max(self.best_loss, self.eval_epoch())

        for self.epoch in range(self.epoch+1, self.nb_epochs+1):

            self.train_loss += [self.train_epoch()]
            self.eval_loss += [self.eval_epoch()]

            if self.epoch % self.save_every == 0:
                self.save_checkpoint(
                    os.path.join(self.odir, f'{self.epoch:02d}.ckpt')
                )
                self.save_checkpoint(
                    os.path.join(self.odir, 'last.ckpt')
                )
            if self.eval_loss[-1] < self.best_loss:
                self.best_loss = self.eval_loss[-1]
                self.save_checkpoint(
                    os.path.join(self.odir, 'best.ckpt')
                )

        self.save_checkpoint(
            os.path.join(self.odir, 'last.ckpt')
        )

    def test(self):
        self.model.eval()
        with torch.no_grad():
            avg_loss = 0
            for batch, (fix, mov) in enumerate(self.testset):
                fix, mov = fix.to(self.device), mov.to(self.device)
                flow = self.model(fix, mov)
                reg = self.energy(flow)
                flow = self.tovalue(flow)
                moved = self.pull(mov, flow)
                sim = self.loss(moved, fix)
                loss = sim + reg

                loss, sim, reg = loss.item(), sim.item(), reg.item()
                avg_loss = (batch * avg_loss + loss) / (batch + 1)

                if batch % 400 == 0 or batch == len(self.testset) - 1:
                    print(f'{self.epoch:02d} | test  | {batch:05d} | '
                          f'{sim:6.3f} + {reg:6.3f} = {loss:6.3f} '
                          f'(average: {avg_loss:6.3f})', end='\r')
        print('')


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', default='.')
    parser.add_argument('--order', type=str, default=3)
    parser.add_argument('--nb-epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save-every', type=int, default=1)
    parser.add_argument('--lam', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--device', default=None)
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--test', default=False, action='store_true')

    prm = parser.parse_args(sys.argv[1:])

    if prm.checkpoint or prm.test:
        cpkt = prm.checkpoint or os.path.join(prm.directory, 'best.ckpt')
        model = PyrTrainer.from_checkpoint(cpkt)
    else:
        model = PyrTrainer(
            odir=prm.directory,
            order=prm.order,
            nb_epochs=prm.nb_epochs,
            lr=prm.lr,
            save_every=prm.save_every,
            lam=prm.lam,
            batch_size=prm.batch_size,
            device=prm.device,
        )

    if prm.test:
        model.test()
    else:
        model.train()
