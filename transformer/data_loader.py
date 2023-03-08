import os
import spacy
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple, List
from collections import Counter

try:
    from transformer.vocab import Vocab
    from transformer.data import load_dataset
except ImportError:
    from .vocab import Vocab
    from .data import load_dataset


UNK_IDX = 0
PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_masks(
    src_batch: torch.Tensor, tgt_batch: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # ----------------------------------------------------------------------
    # [1] padding mask
    # ----------------------------------------------------------------------

    # wherever the padding isn't 1 that's valid
    # (batch_size, 1, max_src_seq_len)
    src_pad_mask = (src_batch != PAD_IDX).unsqueeze(1)

    # (batch_size, 1, max_tgt_seq_len)
    tgt_pad_mask = (tgt_batch != PAD_IDX).unsqueeze(1)

    # ----------------------------------------------------------------------
    # [2] subsequent mask for decoder inputs
    # ----------------------------------------------------------------------
    max_tgt_sequence_length = tgt_batch.shape[1]
    tgt_attention_square = (max_tgt_sequence_length, max_tgt_sequence_length)

    # # full attention
    full_mask = torch.full(tgt_attention_square, 1)

    # # subsequent sequence should be invisible to each token position
    subsequent_mask = torch.tril(full_mask)

    # # add a batch dim (1, max_tgt_seq_len, max_tgt_seq_len)
    subsequent_mask = subsequent_mask.unsqueeze(0)

    return src_pad_mask, tgt_pad_mask & subsequent_mask
    # return src_pad_mask


def generate_and_save_tokens(tokenizer, texts, save_path, most_common=10000):
    counter = Counter()
    for doc in tokenizer.pipe(texts):
        token_texts = []
        for token in doc:
            token_text = token.text.strip()
            if len(token_text) > 0:  # not a white space
                token_texts.append(token_text)
        counter.update(token_texts)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    tokens = [token for token, _ in counter.most_common(most_common)]

    with open(save_path, "w") as f:
        f.writelines("\n".join(tokens))


def make_eng_german_dataloader(
    english_token_path,
    german_token_path,
    batch_size: int,
    device: torch.device,
    split_type: str = "train",
) -> DataLoader:
    de_tokenizer = spacy.load("de_core_news_sm")
    en_tokenizer = spacy.load("en_core_web_sm")

    with open(english_token_path, "r") as f:
        german_tokens = f.read().strip().split("\n")

    with open(german_token_path, "r") as f:
        english_tokens = f.read().strip().split("\n")

    source_vocab = Vocab(de_tokenizer, german_tokens)
    target_vocab = Vocab(en_tokenizer, english_tokens)

    dataset = load_dataset("Multi30k", split_type, language_pair=["de", "en"])

    def collate_fn(batch: List[Tuple[str, str]]):
        source_tokens_list = []
        target_tokens_list = []
        for i, (source_sentence, target_sentence) in enumerate(batch):
            # Tokenization
            source_tokens = source_vocab(source_sentence)
            target_tokens = target_vocab(target_sentence)

            target_tokens = [SOS_IDX] + target_tokens + [EOS_IDX]

            source_tokens_list.append(torch.Tensor(source_tokens).long())
            target_tokens_list.append(torch.Tensor(target_tokens).long())

        source_batch = pad_sequence(
            source_tokens_list, padding_value=PAD_IDX, batch_first=True
        )
        target_batch = pad_sequence(
            target_tokens_list, padding_value=PAD_IDX, batch_first=True
        )
        
        label_batch = target_batch[:, 1:]  # SOS_IDX, ...
        target_batch = target_batch[:, :-1]

        source_mask, target_mask = create_masks(source_batch, target_batch)

        all_batches = [
            source_batch,
            target_batch,
            label_batch,
            source_mask,
            target_mask,
        ]

        # move everything to the target device
        return [x.to(device) for x in all_batches]

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)


if __name__ == "__main__":
    de_tokenizer = spacy.load("de_core_news_sm")
    en_tokenizer = spacy.load("en_core_web_sm")

    with open("dataset/german_tokens.txt", "r") as f:
        german_tokens = f.read().strip().split("\n")

    with open("dataset/english_tokens.txt", "r") as f:
        english_tokens = f.read().strip().split("\n")

    source_vocab = Vocab(de_tokenizer, german_tokens)
    target_vocab = Vocab(en_tokenizer, english_tokens)

    dataset = load_dataset("Multi30k", "train", language_pair=["de", "en"])
    dataloader = make_eng_german_dataloader(
        dataset, source_vocab, target_vocab, 32, device
    )
