import numpy as np
import torch

def hat_so3(v: torch.Tensor):
    Rs = torch.zeros((*v.shape[:-2],3,3), dtype=v.dtype)
    Rs[...,0,1] = -v[...,2,0]
    Rs[...,0,2] = v[...,1,0]
    Rs[...,1,0] = v[...,2,0]
    Rs[...,1,2] = -v[...,0,0]
    Rs[...,2,0] = -v[...,1,0]
    Rs[...,2,1] = v[...,0,0]
    return Rs

def vee_so3(R:torch.Tensor):
    return 0.5 * torch.stack([R[...,2,1]-R[...,1,2],
                              R[...,0,2]-R[...,2,0],
                              R[...,1,0]-R[...,0,1]], axis=-1)

# quaternion norm (adopted from rowan)
def qnorm(q):
    return torch.linalg.norm(q, dim=-1, keepdim=True)

# quaternion sym distance (adopted from rowan)
def qsym_distance(p, q):
    return torch.minimum(qnorm(p - q), qnorm(p + q))

def slice_dataset(x, window_size, stride=None):
    """
    Slice the dataset into windows of size window_size with stride stride.
    The last window will be truncated if there are not enough samples.
    """
    if stride is None:
        stride = window_size - 1
    slices = []
    T, X = x.shape
    N = np.floor((T-window_size)/stride).astype(int) + 1
    slices = np.empty((N, window_size, X))
    for i in range(N):
        slices[i,:,:] = x[i*(window_size-1):(i+1)*window_size-i,:]
    return slices
