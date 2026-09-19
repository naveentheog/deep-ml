import torch

def flatten_then_reshape(x: torch.Tensor, new_shape) -> torch.Tensor:
    return x.flatten().reshape(new_shape)
    # TODO: flatten x to 1-D, then rearrange into new_shape
    pass

def transpose_last_two(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(-2,-1)
    # TODO: swap the last two dimensions of x
    pass
