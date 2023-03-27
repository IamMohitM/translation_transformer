import math
import torch
from .attention import MultiHeadAttentionParrallel
from .position_ffn import PositionWiseFFN
from .residual_add_norm import AddNorm
from .utils import make_clones
from .position import PositionalEncoding

# from d2l import torch as d2l


class EncoderBlock(torch.nn.Module):
    """
    Implements one encode block
    """

    def __init__(
        self,
        num_heads: int,
        num_hidden: int,
        embed_dim: int,
        ffn_hidden: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.attention = MultiHeadAttentionParrallel(
            num_heads=num_heads, embed_dim=embed_dim
        )
        self.position_ffn = PositionWiseFFN(
            ffn_num_hidden=ffn_hidden, ffn_num_output=embed_dim
        )
        self.add_norm_1 = AddNorm(embed_dim, dropout=dropout)
        self.add_norm_2 = AddNorm(embed_dim, dropout=dropout)

    def forward(self, x, valid_lens=None) -> torch.tensor:
        sub_layer_1_output = self.add_norm_2(x, self.attention(x, x, x, valid_lens))

        return self.add_norm_2(
            sub_layer_1_output, self.position_ffn(sub_layer_1_output)
        )


class Encoder(torch.nn.Module):
    def __init__(
        self,
        num_hidden: int,
        num_heads: int,
        embed_dim: int,
        ffn_hidden: int,
        vocab_size: int,
        max_length: int = 1000,
        dropout: float = 0.1,
        num_blocks=6,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim=embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim=embed_dim, max_len=max_length, dropout=dropout)
        self.blocks = torch.nn.Sequential()
        for i in range(num_blocks):
            self.blocks.add_module(
                f"encoder_block_{i+1}",
                EncoderBlock(num_heads, num_hidden, embed_dim, ffn_hidden, dropout),
            )

    def forward(self, X: torch.tensor, valid_lens: torch.tensor = None):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.embed_dim))
        self.attention_weights = [None] * len(self.blocks)
        for i, block in enumerate(self.blocks):
            X = block(X, valid_lens)
            # self.attention_weights[i] = torch.cat(
            #     [head.attention_weights for head in block.attention.attention_heads],
            #     dim=0,
            # )
        return X
    


if __name__ == "__main__":
    num_heads = 8
    embed_dim = 128
    batch_size = 128
    num_kq_v = 8
    ffn_hidden = 1024
    total_words = 1

    dq = 64  # dq = dk = dv = num_hiddens
    # X = torch.rand((batch_size, num_kq_v, embed_dim))
    # encoder_block = EncoderBlock(num_heads=num_heads, num_hidden=dq, embed_dim=embed_dim, ffn_hidden=1024)

    # output = encoder_block(X)
    # print(output.shape)

    encoder = Encoder(
        dq, num_heads, embed_dim, ffn_hidden, num_blocks=2, vocab_size=200
    )
    X = torch.ones((2, total_words, embed_dim), dtype=torch.long)
    encoder_ouptut = encoder(X)
    print(encoder_ouptut.shape)
