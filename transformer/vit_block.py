import torch
from .attention import MultiHeadAttentionParrallel, MultiHeadAttention
from .vit_mlp import ViTMLP


class ViTBlock(torch.nn.Module):
    def __init__(
        self,
        embed_dim,
        norm_shape,
        mlp_num_hiddens,
        num_heads,
        dropout,
        use_bias=False,
    ) -> None:
        super().__init__()

        self.ln1 = torch.nn.LayerNorm(norm_shape)
        self.attention = MultiHeadAttentionParrallel(
            num_heads=num_heads,
            embed_dim=embed_dim,
            bias=use_bias,
            dropout=dropout
        )
        self.ln2 = torch.nn.LayerNorm(norm_shape)
        self.mlp = ViTMLP(
            mlp_num_hiddens=mlp_num_hiddens, mlp_num_outputs=embed_dim, dropout=dropout
        )

    def forward(self, X, valid_lens=None):
        # (the star is unpacking normed input)
        X = X + self.attention(*([self.ln1(X)] * 3), valid_lens)
        return X + self.mlp(self.ln2(X))


class VitBlockSequential(ViTBlock):
    def __init__(self, embed_dim, norm_shape, mlp_num_hiddens, num_heads, dropout, use_bias=False) -> None:
        super().__init__(embed_dim, norm_shape, mlp_num_hiddens, num_heads, dropout, use_bias)
        
        assert embed_dim % num_heads == 0
        num_hidden = embed_dim // num_heads
        self.attention = MultiHeadAttention(num_heads = num_heads, num_hidden=num_hidden, embed_dim=embed_dim, bias = use_bias)