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
