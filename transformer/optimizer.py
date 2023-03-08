"""_summary_

Credits: https://nlp.seas.harvard.edu/2018/04/03/attention.html#decoder

"""

import math
import torch


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(
        self,
        embed_dim,
        optimizer,
        warmup=1000,
    ):
        self.optimizer : torch.optim.Optimizer= optimizer
        self._step = 0
        self.warm_up_steps = warmup
        self.embed_dim = embed_dim
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.zero_grad()
        self.optimizer.step()

    def rate(self, step_num=None):
        "Implement `lrate` above"
        if step_num is None:
            step_num = self._step

        return math.pow(self.embed_dim, -0.5) * min(
            math.pow(step_num, -0.5), step_num * math.pow(self.warm_up_steps, -1.5)
        )
