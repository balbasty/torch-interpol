
import torch


def saveargs(func):

    def wrapper(self, *args, **kwargs):
        out = func(self, *args, **kwargs)
        self._args = args
        self._kwargs = kwargs
        return out

    return wrapper


class LoadableModule(torch.nn.Module):

    def save_checkpoint(self, fname=None, **more_stuff):
        ckpt = dict(
            args=self._args,
            kwargs=self._kwargs,
            state=self.state_dict(),
            module=type(self).__module__,
            name=type(self).__name__,
            **more_stuff,
        )
        if fname:
            torch.saveckpt(ckpt, fname)
        return ckpt

    def load_checkpoint(self, ckpt):
        if not isinstance(ckpt, dict):
            ckpt = torch.load(ckpt)
        self.load_state_dict(ckpt.state)
        more_stuff = {
            key: val for key, val in ckpt.items()
            if key not in ('args', 'kwargs', 'state', 'module', 'name')
        }
        self.load_more_stuff(**more_stuff)
        return self

    def load_more_stuff(self, **more_stuff):
        pass

    @classmethod
    def from_checkpoint(cls, ckpt):
        ckpt = torch.load(ckpt)
        obj = cls(*ckpt.args, **ckpt.kwargs)
        obj.load_checkpoint(ckpt)
        return obj
