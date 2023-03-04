import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

d2l.use_svg_display()

def nadaraya_watson(x_train, y_train, x_val, kernel):
    dists = x_train.reshape((-1, 1)) - x_val.reshape((1, -1))
    # Each column/row corresponds to each query/key
    k = kernel(dists).type(torch.float32)
    attention_w = k / k.sum(0)  # Normalization over keys for each query
    y_hat = y_train@attention_w
    return y_hat, attention_w

def f(x):
    return 2 * torch.sin(x) + x

# Define some kernels
def gaussian(x):
    return torch.exp(-x**2 / 2)

def boxcar(x):
    return torch.abs(x) < 1.0

def constant(x):
    return 1.0 + 0 * x

def epanechikov(x):
    return torch.max(1 - torch.abs(x), torch.zeros_like(x))

if __name__ == "__main__":

    n = 40
    x_train, _ = torch.sort(torch.rand(n) * 5)
    y_train = f(x_train) + torch.randn(n)
    x_val = torch.arange(0, 5, 0.1)
    y_val = f(x_val)

    fig, axes = d2l.plt.subplots(1, 4, sharey=True, figsize=(12, 3))

    kernels = (gaussian, boxcar, constant, epanechikov)
    names = ('Gaussian', 'Boxcar', 'Constant', 'Epanechikov')

    x = torch.arange(-2.5, 2.5, 0.1)
    for kernel, name, ax in zip(kernels, names, axes):
        ax.plot(x.detach().numpy(), kernel(x).detach().numpy())
        ax.set_xlabel(name)
