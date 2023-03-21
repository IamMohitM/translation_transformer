import math
import torch
from torch import nn
from d2l import torch as d2l
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def masked_softmax(X, valid_lens):  # @save
    """Perform softmax operation by masking elements on the last axis."""

    # X: 3D tensor, valid_lens: 1D or 2D tensor
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.size(1)
        mask = (
            torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :]
            < valid_len[:, None]
        )
        X[~mask] = value
        return X

    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class DotProductAttention(nn.Module):  # @save
    """Scaled dot product attention."""

    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Swap the last two dimensions of keys with keys.transpose(1, 2)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class MultiHeadAttention(d2l.Module):  # @save
    """Multi-head attention."""

    def __init__(
        self,
        num_hiddens: int,
        num_heads: int,
        dropout: int,
        bias: bool = False,
        **kwargs
    ):
        super().__init__()
        self.num_heads = num_heads
        # self.attention = d2l.DotProductAttention(dropout)
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_k = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_v = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_o = nn.LazyLinear(num_hiddens, bias=bias)

    def forward(
        self,
        queries: torch.tensor,
        keys: torch.tensor,
        values: torch.tensor,
        valid_lens: torch.tensor,
    ):
        # Shape of queries, keys, or values:
        # (batch_size, no. of queries or key-value pairs, num_hiddens)
        # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
        # After transposing, shape of output queries, keys, or values:
        # (batch_size * num_heads, no. of queries or key-value pairs,
        # num_hiddens / num_heads)
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for num_heads
            # times, then copy the next item, and so on
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0
            )

        # Shape of output: (batch_size * num_heads, no. of queries,
        # num_hiddens / num_heads)
        output = self.attention(queries, keys, values, valid_lens)
        # Shape of output_concat: (batch_size, no. of queries, num_hiddens)
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat)


@d2l.add_to_class(MultiHeadAttention)  # @save
def transpose_qkv(self, X):
    """Transposition for parallel computation of multiple attention heads."""
    # Shape of input X: (batch_size, no. of queries or key-value pairs,
    # num_hiddens). Shape of output X: (batch_size, no. of queries or
    # key-value pairs, num_heads, num_hiddens / num_heads)
    X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
    # Shape of output X: (batch_size, num_heads, no. of queries or key-value
    # pairs, num_hiddens / num_heads)
    X = X.permute(0, 2, 1, 3)
    # Shape of output: (batch_size * num_heads, no. of queries or key-value
    # pairs, num_hiddens / num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])


@d2l.add_to_class(MultiHeadAttention)  # @save
def transpose_output(self, X):
    """Reverse the operation of transpose_qkv."""
    X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


if __name__ == "__main__":
    # embedding = torch.rand((3, 1, 128), device="cuda")
    # embedding_batch_size, dim = embedding.shape[0], embedding.shape[-1]
    # query_input, key_input, value_input = embedding.repeat(
    #     embedding_batch_size, 1, 1
    # ).reshape(embedding.shape[0], 3, embedding.shape[-1])
    # model = MultiHeadAttention(64, 8, 0.5).to(device)
    # output = model(query_input, key_input, value_input, valid_lens = [128, 128, 128])
    # # attention_head = AttentionHead(False)
    # # output = attention_head(emb)
    # print(output.shape)

    # num_hiddens, num_heads = 64, 4
    # attention = MultiHeadAttention(num_hiddens, num_heads, 0.5)
    # attention.to(device)
    # batch_size, num_queries, num_kvpairs = 2, 4, 6
    # valid_lens = torch.tensor([3, 2], device = device)
    # X = torch.ones((batch_size, num_queries, 10), device= device)
    # Y = torch.ones((batch_size, num_kvpairs, 10), device = device)
    # output = (attention(X, Y, Y, valid_lens))
    # d2l.check_shape(output,
    #                 (batch_size, num_queries, num_hiddens))

    # encoder = d2l.TransformerEncoder(200, 24, 48, 8, 2, 0.5)
    encoder = d2l.TransformerEncoder(200, 10, 48, 8, 2, 0.5)
    output = encoder(torch.ones((2, 20), dtype=torch.long), valid_lens=None)
    d2l.check_shape(output, (2, 100, 24), (2, 100, 24))
