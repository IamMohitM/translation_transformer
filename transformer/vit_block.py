import torch
from .attention import MultiHeadAttention
from .vit_mlp import ViTMLP


class ViTBlock(torch.nn.Module):
    def __init__(
        self,
        embed_dim,
        num_hiddens,
        norm_shape,
        mlp_num_hiddens,
        num_heads,
        dropout,
        use_bias=False,
    ) -> None:
        super().__init__()

        self.ln1 = torch.nn.LayerNorm(norm_shape)
        self.attention = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_hidden=num_hiddens,
            bias=use_bias,
        )
        self.ln2 = torch.nn.LayerNorm(norm_shape)
        self.mlp = ViTMLP(
            mlp_num_hiddens=mlp_num_hiddens, mlp_num_outputs=embed_dim, dropout=dropout
        )

    def forward(self, X, valid_lens=None):
        # (the star is unpacking normed input)
        X = X + self.attention(*([self.ln1(X)] * 3), valid_lens)
        return X + self.mlp(self.ln2(X))
