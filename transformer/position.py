import torch


class PositionalEncoding(torch.nn.Module):
    def __init__(self, embed_dim, dropout=0, max_len=1000) -> None:
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)

        self.P = torch.zeros((1, max_len, embed_dim))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, embed_dim, 2, dtype=torch.float32) / embed_dim
        )

        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, : X.shape[1], :].to(X.device)
        return self.dropout(X)


if __name__ == "__main__":
    encoding_dim, num_steps = 32, 60
    pos_encoding = PositionalEncoding(encoding_dim, 0)
    X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
    P = pos_encoding.P[:, : X.shape[1], :]
    # d2l.plot(torch.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
    #         figsize=(6, 2.5), legend=["Col %d" % d for d in torch.arange(6, 10)])
