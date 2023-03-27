import torch
from .patch_embedding import PatchEmbedding
from .vit_block import VitBlockSequential, ViTBlock


class ViT(torch.nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        embed_dim,
        mlp_num_hidden,
        num_heads,
        num_blocks,
        emb_dropout,
        block_dropout,
        use_bias=False,
        num_classes=10,
        **kwargs,
    ) -> None:
        super().__init__()
        self.patch_embedding = PatchEmbedding(
            img_size, patch_size, num_hiddens=embed_dim
        )
        self.class_token = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))

        num_steps = self.patch_embedding.num_patches + 1  # +1 because of class_token
        self.position_embedding = torch.nn.Parameter(
            torch.randn(1, num_steps, embed_dim)
        )

        self.dropout = torch.nn.Dropout(emb_dropout)
        self.blocks = torch.nn.Sequential()
        for block in range(num_blocks):
            self.blocks.add_module(
                f"Block {block}",
                ViTBlock(
                    embed_dim=embed_dim,
                    norm_shape=embed_dim,
                    mlp_num_hiddens=mlp_num_hidden,
                    num_heads=num_heads,
                    dropout=block_dropout,
                    use_bias=use_bias,
                ),
            )

        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(embed_dim), torch.nn.Linear(embed_dim, num_classes)
        )

    def update_attention(self, block_num: int, block):
        self.attention_weights[block_num] = block.attention.attention.attention_weights

    def forward(self, X):
        X = self.patch_embedding(X)
        X = torch.cat((self.class_token.expand(X.shape[0], -1, -1), X), 1)
        X = self.dropout(X + self.position_embedding)
        self.attention_weights = [None] * len(self.blocks)
        for block_num, block in enumerate(self.blocks):
            X = block(X)
            self.update_attention(block_num, block)
        return self.head(X[:, 0])  # gets only latent embedding


class ViTSequential(ViT):
    def __init__(
        self,
        img_size,
        patch_size,
        embed_size,
        mlp_num_hiddens,
        num_heads,
        num_blocks,
        emb_dropout,
        block_dropout,
        use_bias=False,
        num_classes=10,
    ) -> None:
        super().__init__(
            img_size,
            patch_size,
            embed_size,
            mlp_num_hiddens,
            num_heads,
            num_blocks,
            emb_dropout,
            block_dropout,
            use_bias,
            num_classes,
        )

        for block in range(num_blocks):
            self.blocks.add_module(
                f"Block {block}",
                VitBlockSequential(
                    embed_dim=embed_size,
                    norm_shape=embed_size,
                    mlp_num_hiddens=mlp_num_hiddens,
                    num_heads=num_heads,
                    dropout=block_dropout,
                    use_bias=use_bias,
                ),
            )

    def update_attention(self, block_num: int, block):
        ...
