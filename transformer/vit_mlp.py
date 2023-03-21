import torch

class ViTMLP(torch.nn.Module):

    def __init__(self, mlp_num_hiddens: int, mlp_num_outputs: int, dropout=0.5) -> None:
        super().__init__()
        self.dense1 = torch.nn.LazyLinear(mlp_num_hiddens)
        self.gelu = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(dropout)
        self.dense2 = torch.nn.LazyLinear(mlp_num_outputs)
        self.dropout2 = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.gelu(self.dense1(x)))
        return self.dropout2(self.gelu(self.dense2(x)))
