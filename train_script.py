import torch

from transformer.transformer import EncoderDecoder
from transformer.data_loader import make_eng_german_dataloader
from transformer.optimizer import NoamOpt

##param definition
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_heads = 8
embed_dim = 128
batch_size = 32
ffn_num_hidden = 256
total_words = 4
num_hidden = 64  # dq = dk = dv = num_hiddens
num_blocks = 2
dropout = 0.1
vocab_size = 12004
warmup = 1000

num_epochs = 10


german_tokens_path = "dataset/german_tokens.txt"
english_tokens_path = "dataset/english_tokens.txt"

valid_english_tokens_path = "dataset/german_tokens"
# model dev


def get_model(embed_dim, num_hidden, ffn_num_hidden, num_heads, num_blocks, vocab_size):
    return EncoderDecoder(
        embed_dim=embed_dim,
        num_hidden=num_hidden,
        ffn_hidden=ffn_num_hidden,
        num_heads=num_heads,
        num_blocks=num_blocks,
        vocab_size=vocab_size,
    ).to(device)


def get_dataloader(english_tokens_path, german_tokens_path, split_type, batch_size):
    return make_eng_german_dataloader(
        english_tokens_path,
        german_tokens_path,
        batch_size,
        device,
        split_type=split_type,
    )


def training_loop():
    ...


if __name__ == "__main__":
    model = get_model(
        embed_dim, num_hidden, ffn_num_hidden, num_heads, num_blocks, vocab_size
    )
    train_dataloader = get_dataloader(
        english_tokens_path, german_tokens_path, "train", batch_size
    )
    val_dataloader = get_dataloader(
        english_tokens_path, german_tokens_path, "valid", batch_size
    )
    test_dataloader = get_dataloader(
        english_tokens_path, german_tokens_path, "test", batch_size
    )

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, betas = [0.9, 0.98], eps = -1e9)
    optimizer = NoamOpt(embed_dim, optimizer, warmup=warmup)
    

    for epoch in range(num_epochs):
        
        ...
    

