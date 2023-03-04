import torch

class PositionWiseFFN(torch.nn.Module):
    def __init__(self, ffn_num_hidden, ffn_num_output) -> None:
        super().__init__()
        self.dense1 = torch.nn.LazyLinear(ffn_num_hidden)
        self.relu = torch.nn.ReLU()
        self.dense2 = torch.nn.Linear(self.dense1.out_features, ffn_num_output)

    def forward(self, x):
        return self.dense2(self.relu(self.dense1(x)))