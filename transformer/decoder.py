from typing import Tuple

import math
import torch
from .attention import MultiHeadAttention
from .residual_add_norm import AddNorm
from .position_ffn import PositionWiseFFN
from .position import PositionalEncoding


class DecoderBlock(torch.nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_hidden: int,
        embed_dim: int,
        ffn_hidden: int,
        i: int,
        bias=True,
        dropout=0.1,
    ) -> None:
        super().__init__()
        self.i = i
        self.masked_multihead_attention = MultiHeadAttention(
            num_heads, num_hidden, embed_dim, bias=bias
        )
        self.add_norm_1 = AddNorm(embed_dim, dropout)
        self.attention = MultiHeadAttention(num_heads, num_hidden, embed_dim, bias=bias)
        self.add_norm_2 = AddNorm(embed_dim, dropout)
        self.ffn = PositionWiseFFN(ffn_hidden, embed_dim)
        self.add_norm_3 = AddNorm(embed_dim, dropout)

    def forward(self, X, state, dec_valid_lens = None):
        # encoder inputs
        enc_outputs, enc_valid_lens = state[0], state[1]

        # num_steps is usually the maximum sequence length
        # We only edit index 2 of state because encoder inputs are same for all blocks
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), dim=1)
        state[2][self.i] = key_values

        #get the mask for training
        # if self.training:
        #     batch_size, num_steps, _ = X.shape
        #     # Shape of dec_valid_lens: (batch_size, num_steps), where every
        #     # row is [1, 2, ..., num_steps]
        #     dec_valid_lens = torch.arange(1, num_steps + 1, device=X.device).repeat(
        #         batch_size, 1
        #     )
        # else:
        #     dec_valid_lens = None

        # masked Self-attention
        X2 = self.masked_multihead_attention(X, key_values, key_values, dec_valid_lens)
        Y = self.add_norm_1(X, X2)
        # Encoder-decoder attention. Shape of enc_outputs:
        # (batch_size, num_steps, embed_dim)
        # keys and values from encoder
        Y2 = self.attention(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.add_norm_2(Y, Y2)
        return self.add_norm_3(Z, self.ffn(Z)), state


# http://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html
class Decoder(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_heads: int,
        num_hidden: int,
        embed_dim: int,
        ffn_hidden: int,
        num_blocks: int,
        max_length: int = 1000,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.num_blocks = num_blocks
        self.embed_dim = embed_dim
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim=embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim=embed_dim, max_len=max_length)
        self.decoder_sub = torch.nn.Sequential()
        for i in range(self.num_blocks):
            self.decoder_sub.add_module(
                f"DecoderBlock_{i}",
                DecoderBlock(
                    num_heads, num_hidden, embed_dim, ffn_hidden, i, dropout=dropout
                ),
            )
        self.dense = torch.nn.LazyLinear(vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.num_blocks]

    def forward(self, X, state, decoder_valid_lens = None) -> Tuple:
        # Add position embeddings
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.embed_dim))
        # 2 because there are two attention layers in each block
        self._attention_weights = [[None] * len(self.decoder_sub) for _ in range(2)]

        for i, block in enumerate(self.decoder_sub):
            X, state = block(X, state, decoder_valid_lens)
            self._attention_weights[0][i] = torch.cat(
                [
                    head.attention_weights
                    for head in block.masked_multihead_attention.attention_heads
                ],
                dim=0,
            )

            self._attention_weights[1][i] = torch.cat(
                [head.attention_weights for head in block.attention.attention_heads],
                dim=0,
            )

        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights
