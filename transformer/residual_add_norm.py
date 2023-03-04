import torch


class AddNorm(torch.nn.Module):
    def __init__(self, norm_shape, dropout) -> None:
        super().__init__()
        self.ln = torch.nn.LayerNorm(norm_shape)
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, x, y):
        return self.ln(self.dropout(y) + x)