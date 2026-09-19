import torch

def to_float_tensor(values):
    x=torch.tensor(values, dtype=torch.float32)
    return x
    # TODO: return a torch.float32 tensor built from `values`
    pass
