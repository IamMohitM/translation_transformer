import torch
from position_ffn import PositionWiseFFN

class FFN(torch.nn.Module):
    def __init__(self, pos_ffn: PositionWiseFFN, layer_norm_shape: tuple, dropout: float) -> None:
        super().__init__()
        self.position_ffn = pos_ffn
        self.layer_norm = torch.nn.LayerNorm(layer_norm_shape)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        return self.layer_norm(x + self.dropout(self.position_ffn(x)))
    

if __name__ == "__main__":
    q = torch.randn(100).reshape(1, -1)
    layer_norm = torch.nn.LayerNorm(100)
    model = FFN(layer_norm)
    output = model(q)
    print(output.shape)