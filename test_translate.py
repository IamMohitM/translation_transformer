from transformer.translate import translate
from transformer.transformer import EncoderDecoder
from transformer.prepare_data import build_vocab, get_dataset_iter, load_tokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_heads = 8
    embed_dim = 512
    ffn_num_hidden = 512
    num_hidden = 64  # dq = dk = dv = num_hiddens
    num_blocks = 6
    dropout = 0.1

    dataset = get_dataset_iter("train", ["de", "en"])

    source_tokenizer = load_tokenizer("de")
    target_tokenizer = load_tokenizer("en")
    source_vocab_path = "checkpoints/source_vocab.pth"
    target_vocab_path = "checkpoints/target_vocab.pth"
    source_vocab = torch.load(source_vocab_path)
    target_vocab = torch.load(target_vocab_path)
    # source_vocab = build_vocab(source_tokenizer, dataset, "de")
    # target_vocab = build_vocab(target_tokenizer, dataset, "en")

    model = EncoderDecoder(
        embed_dim=embed_dim,
        num_hidden=num_hidden,
        ffn_hidden=ffn_num_hidden,
        num_heads=num_heads,
        num_blocks=num_blocks,
        source_vocab_size=len(source_vocab),
        target_vocab_size=len(target_vocab),
    ).to(device)

    checkpoint_path = "checkpoints/model.pt"

    model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
    model.eval()

    print(
        translate(
            model,
            source_vocab,
            source_tokenizer,
            target_vocab,
            target_tokenizer,
            # "Ein schwarzer Hund und ein gefleckter Hund kämpfen.",
            "Vier Typen, von denen drei Hüte tragen und einer nicht, springen oben in einem Treppenhaus.",
        )
    )
