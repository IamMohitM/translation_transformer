import torch


class PatchEmbedding(torch.nn.Module):
    def __init__(
        self, img_size=96, patch_size=16, in_channels=1, num_hiddens=512
    ) -> None:
        super().__init__()

        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x

        img_size, patch_size = _make_tuple(img_size), _make_tuple(patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[0]
        )
        self.conv = torch.nn.Conv2d(
            in_channels, num_hiddens, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, X):
        # output -  (batch, num_paches, num_channels)
        return self.conv(X).flatten(2).transpose(1, 2)


if __name__ == "__main__":
    img_size, patch_size, num_hiddens, batch_size = 96, 16, 512, 4
    patch_emb = PatchEmbedding(img_size, patch_size, num_hiddens)
    X = torch.rand(batch_size, 3, img_size, img_size)
    X = patch_emb(X)
    print(X.shape)
