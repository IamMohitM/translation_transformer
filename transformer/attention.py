import torch
import math

from .utils import make_clones


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def masked_softmax(X, valid_lens):  # @save
    """Perform softmax operation by masking elements on the last axis."""

    # X: 3D tensor, valid_lens: 1D or 2D tensor
    def _sequence_mask(X, valid_len, value=0):
        # if same shape - apply directly
        if X.shape == valid_len.shape:
            X[~valid_len] = value
        else:
            maxlen = X.size(1)
            # broadcasting here
            # first element will be 1 * maxlen
            # second elemnt is len(valid_len) * 1
            # mask will be len(valid_len) * maxlen
            mask = (
                torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :]
                < valid_len[:, None]
            )
            # The mask describes which query and pair are invalid
            X[~mask] = value
        return X

    if valid_lens is None:
        return torch.nn.functional.softmax(X, dim=-1)
    else:
        # TODO: optimize masking
        shape = X.shape
        if valid_lens.dim() == 1:
            # repeat them for each token in X
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        elif valid_lens.dim() == 3:
            # if one mask for each batch
            if valid_lens.shape[1] == 1:
                # (batch, 1, maxlen)
                valid_lens = torch.sum(valid_lens, axis=2).squeeze()
        else:
            valid_lens = valid_lens.reshape(-1)

        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = _sequence_mask(X, valid_lens, value=-1e6)
        return torch.nn.functional.softmax(X.reshape(shape), dim=-1)


class AttentionHead(torch.nn.Module):
    """Implements one attention head

    Parameters
    ----------
    torch : _type_
        _description_
    """

    def __init__(self, out_dim=64, bias=True, dropout=0.1) -> None:
        super().__init__()
        # Define one linear layer for each m
        self.w_q = torch.nn.LazyLinear(out_dim, bias=bias)
        self.w_k = torch.nn.LazyLinear(out_dim, bias=bias)
        self.w_v = torch.nn.LazyLinear(out_dim, bias=bias)
        # TODO: Need to check where relu needs to be used
        # self.output = torch.nn.LazyLinear(512, bias=bias)

    def forward(
        self,
        queries: torch.tensor,
        keys: torch.tensor,
        values: torch.tensor,
        valid_lens: torch.tensor = None,
    ):
        Query = self.w_q(queries)
        Key = self.w_k(keys)
        Value = self.w_v(values)

        dim = Query.shape[-1]
        # Q shape - (batch, num_queires, dim) -each dim size descibes each query - if dim = 10 - then each query in num_q is represented by 10 numbers
        # Key shape - (batch, num_kv_pairs, dim) - changed to (batch, dim, num_kv_paris) - each dim size descibes each key
        # scores (batch, num_queries, num_kv_pairs) - describes score for each query to all 6 keys
        # the bottom performs a dot product between each query and each key - highest value means the query and key are similar
        scores = torch.bmm(Query, Key.transpose(-2, -1)) / math.sqrt(dim)
        # attention weights are multiplied to the values
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.attention_weights, Value)


class DotProductAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def masked_softmax(X, valid_lens):
        # X shape is 3D (batch, tokens, token)
        def sequence_mask(X, valid_len, value=-1e6):
            if X.shape == valid_len.shape:
                X[~valid_len] == value
            else:
                # torch.arange()
                maxlen = X.shape[1]
                mask = (
                    torch.arange((maxlen), dtype=torch.float32, device=X.device)[
                        None, :
                    ]
                    < valid_len[:, None]
                )
                X[~mask] = value
            return X

        if not valid_lens is None:
            # valid_lens can be 1D, 2D, 3D
            # if 1D, then each value represents max value for a batch
            if valid_lens.dim() == 1:
                ...
            X = sequence_mask(X, valid_lens, -1e6)

        return torch.nn.functional.softmax(X, dim=-1)

    def forward(self, queries, keys, values, valid_lens=None):
        dim = queries.size(-1)
        # queries - keys - values - (batch, tokens, projected_dims)
        scores = torch.bmm(queries, torch.transpose(keys, 1, 2)) / math.sqrt(dim)
        # scores - batch, tokens, tokens
        # valid lens - None
        # valid lens - 1D
        self.attention_weights = DotProductAttention.masked_softmax(scores, valid_lens)

        # result - (batch, tokens, tokens) * (batch, token, projected_dims) -> (b, t, projected_dims)
        return torch.bmm(self.attention_weights, values)


class MultiHeadAttentionParrallel(torch.nn.Module):
    def __init__(self, num_heads, embed_dim, dropout=0.2, bias=False) -> None:
        super().__init__()
        self.num_heads = num_heads

        # if num_hiddens is None or num_hiddens % num_heads != 0:
        #     num_hiddens = embed_dim
        assert (
            embed_dim % num_heads == 0
        ), f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        self.attention = DotProductAttention()

        self.W_q = torch.nn.Linear(embed_dim, embed_dim, bias)
        self.W_k = torch.nn.Linear(embed_dim, embed_dim, bias)
        self.W_v = torch.nn.Linear(embed_dim, embed_dim, bias)
        self.dropout = torch.nn.Dropout(dropout)

        self.output_linear = torch.nn.Linear(embed_dim, embed_dim, bias)

    def transpose_qkv(self, X):
        # heads divided
        # input - (batch, tokens, dims) - output - (batch, tokens, heads, dim/heads)
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
        # (batch, heads, tokens, dim/heads)
        X = X.permute(0, 2, 1, 3)

        # (batch * heads, tokens, dim/heads)
        return X.reshape(-1, X.shape[-2], X.shape[-1])

    def transpose_output(self, X):
        # reverse of transpose_qkv
        # input (batch * heads, tokens, dim/heads) -> (batch, heads, tokens, dim/heads)
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        # (batch, tokens, heads, dim/heads)
        X = X.permute(0, 2, 1, 3)
        # (batch, tokens, dim)
        return X.reshape(X.shape[0], X.shape[1], -1)

    def forward(self, queries, keys, values, valid_lens=None):
        Queries = self.transpose_qkv(self.W_q(queries))
        Keys = self.transpose_qkv(self.W_k(keys))
        Values = self.transpose_qkv(self.W_v(values))

        attention_output = self.attention(Queries, Keys, Values, valid_lens)

        return self.dropout(self.output_linear(self.transpose_output(attention_output)))


class MultiHeadAttention(torch.nn.Module):
    """
    Implements multi head attention
    input - (batch_size, num_queries_k_v, embed_dim)
    output - (batch_size, num_queries, embed-dim)
    """

    def __init__(self, num_heads, num_hidden, embed_dim, bias=False) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.attention_heads: torch.nn.ModuleList = make_clones(
            AttentionHead, self.num_heads, num_hidden
        )
        # output must be same as embedding dimension size
        self.output = torch.nn.LazyLinear(embed_dim, bias)

    def forward(self, queries, keys, values, valid_lens=None):
        # recieve query - head keys and values (same embedding)
        # process each pair with separate head
        head_inputs = [(queries, keys, values, valid_lens)] * self.num_heads

        # concat output from each head
        outputs = torch.cat(
            [
                attn_head(query, key, value, valid_lens)
                for attn_head, (query, key, value, valid_lens) in zip(
                    self.attention_heads, head_inputs
                )
            ],
            dim=-1,
            # concat over the encoding dimension
        )

        #
        output = self.output(outputs)
        return output


# https://pytorch.org/docs/stable/notes/broadcasting.html
if __name__ == "__main__":
    num_hidden, num_heads = 64, 8
    num_queries = 1  # for each batch
    num_kv_pairs = 1  # for each batch
    batch_size = 1
    embed_dim = 512
    # input dim - (batch_size, num_quereis, embed_dim)
    queries = torch.rand((batch_size, num_queries, embed_dim), device=device)
    keys = torch.rand((batch_size, num_kv_pairs, embed_dim), device=device)
    values = torch.rand((batch_size, num_kv_pairs, embed_dim), device=device)
    # valid_lens = torch.tensor([1, 2], device = device)
    valid_lens = None
    model = MultiHeadAttention(num_heads, num_hidden, embed_dim).to(device)
    output = model(queries, keys, values, valid_lens=valid_lens)
    print(output.shape)
