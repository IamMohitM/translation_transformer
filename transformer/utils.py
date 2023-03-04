import torch


def make_clones(module: torch.nn.Module, copies: int, *args):
    return torch.nn.ModuleList([module(*args) for _ in range(copies)])
