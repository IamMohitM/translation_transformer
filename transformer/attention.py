import torch
import math
import copy

from .utils import make_clones


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def masked_softmax(X, valid_lens):  # @save
    """Perform softmax operation by masking elements on the last axis."""

    # X: 3D tensor, valid_lens: 1D or 2D tensor
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.size(1)
        # broadcasting here
        # first element will be 1 * maxlen
        # second elemnt is len(valid_len) * 1
        # mask will be len(valid_len) * maxlen
        mask = (
            
            torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :]< valid_len[:, None]
        )
        # The mask describes which query and pair are
        X[~mask] = value
        return X

    if valid_lens is None:
        return torch.nn.functional.softmax(X, dim=-1)
    else:
        # TODO: optimize masking
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        elif valid_lens.dim() == 3:
            # if one mask for each batch
            if valid_lens.shape[1] == 1:
                #(batch, 1, maxlen)
                valid_lens = torch.sum(valid_lens, axis = 2).squeeze()
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        try:
            #if same shape - apply directly
            X[~valid_lens] = -1e6
        except IndexError:
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
        self.relu = torch.nn.ReLU()
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
        scores = torch.bmm(Query, Key.transpose(1, 2)) / math.sqrt(dim)
        # attention weights are multiplied to the values
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.matmul(self.attention_weights, Value)


class MultiHeadAttention(torch.nn.Module):
    """
    Implements multi head attention
    input - (batch_size, num_queries_k_v, embed_dim)
    output - (batch_size, num_queries, embed-dim)
    """

    def __init__(self, num_heads, num_hidden, embed_dim, bias=False) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.attention_heads:torch.nn.ModuleList  = make_clones(AttentionHead, self.num_heads, num_hidden)
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
