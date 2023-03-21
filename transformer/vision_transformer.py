import torch
from .patch_embedding import PatchEmbedding
from .vit_block import ViTBlock

class ViT(torch.nn.Module):
    def __init__(self, img_size, patch_size, embed_size, num_hiddens, mlp_num_hiddens, num_heads, num_blocks, emb_dropout, block_dropout, use_bias=False, num_classes = 10) -> None:
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, num_hiddens=embed_size)
        self.class_token = torch.nn.Parameter(torch.zeros(1, 1, embed_size))

        num_steps = self.patch_embedding.num_patches + 1 # +1 because of class_token
        self.position_embedding = torch.nn.Parameter(torch.randn(1, num_steps, embed_size))

        self.dropout = torch.nn.Dropout(emb_dropout)
        self.blocks = torch.nn.Sequential()
        for block in range(num_blocks):
            self.blocks.add_module(f"Block {block}", ViTBlock(embed_dim=embed_size, num_hiddens=num_hiddens, norm_shape=embed_size, mlp_num_hiddens=mlp_num_hiddens, num_heads=num_heads, dropout=block_dropout, use_bias=use_bias))
        
        self.head = torch.nn.Sequential(torch.nn.LayerNorm(embed_size), torch.nn.Linear(embed_size, num_classes))

    def forward(self, X):
        X = self.patch_embedding(X)
        X = torch.cat((self.class_token.expand(X.shape[0], -1, -1), X), 1)
        X = self.dropout(X + self.position_embedding)
        for block in self.blocks:
            X = block(X)
        return self.head(X[:, 0])
    

