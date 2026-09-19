import torch

def grad_of_quadratic(x_value) -> float:
    x = torch.tensor(float(x_value), requires_grad=True)  # float leaf, tracked by autograd
    f = x**2 + 3*x + 2                                    # forward pass records the graph
    f.backward()                                          # backprop from scalar f
    return x.grad.item()                                  # df/dx stored on the leaf