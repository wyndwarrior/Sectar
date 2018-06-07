import torch
import torch.autograd

"""
GPU wrappers
"""

_use_gpu = True


def set_gpu_mode(mode):
    global _use_gpu
    _use_gpu = mode


def gpu_enabled():
    return _use_gpu


# noinspection PyPep8Naming
def DoubleTensor(*args, **kwargs):
    if _use_gpu:
        return torch.cuda.DoubleTensor(*args, **kwargs)
    else:
        # noinspection PyArgumentList
        return torch.DoubleTensor(*args, **kwargs)

def FloatTensor(*args, **kwargs):
    if _use_gpu:
        return torch.cuda.FloatTensor(*args, **kwargs)
    else:
        # noinspection PyArgumentList
        return torch.FloatTensor(*args, **kwargs)

def LongTensor(*args, **kwargs):
    if _use_gpu:
        return torch.cuda.LongTensor(*args, **kwargs)
    else:
        # noinspection PyArgumentList
        return torch.LongTensor(*args, **kwargs)

def Variable(tensor, **kwargs):
    if _use_gpu and not tensor.is_cuda:
        return torch.autograd.Variable(tensor.cuda(), **kwargs)
    else:
        return torch.autograd.Variable(tensor, **kwargs)


def from_numpy(*args, **kwargs):
    if _use_gpu:
        return torch.from_numpy(*args, **kwargs).cuda()
    else:
        return torch.from_numpy(*args, **kwargs)


def get_numpy(tensor):
    if isinstance(tensor, torch.autograd.Variable):
        return get_numpy(tensor.data)
    if _use_gpu:
        return tensor.cpu().numpy()
    return tensor.numpy()


def np_to_var(np_array):
    return Variable(from_numpy(np_array).float())
