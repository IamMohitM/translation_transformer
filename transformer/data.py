import os
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
from torchtext import datasets
from typing import Tuple, List
import spacy
from collections import Counter
import torch
from torch.nn.utils.rnn import pad_sequence
from transformer.vocab import Vocab


UNK_IDX = 0
PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def load_dataset(
    name: str, split: str, language_pair: Tuple[str, str]
) -> IterableDataset:
    dataset_class = eval(f"datasets.{name}")
    dataset = dataset_class(split=split, language_pair=language_pair)
    return dataset


if __name__ == "__main__":
    train_dataset = load_dataset("Multi30k", "train", language_pair=["de", "en"])
    loader = DataLoader(train_dataset, batch_size=1)

    de_tokenizer = spacy.load("de_core_news_sm")
    en_tokenizer = spacy.load("en_core_web_sm")

    german_texts = []
    english_texts = []

    for de_text, en_text in train_dataset:
        german_texts.append(de_text)
        english_texts.append(en_text)

    print(len(german_texts))
    print(len(english_texts))

    generate_and_save_tokens(
        en_tokenizer, english_texts, "dataset/english_tokens.txt", 12000
    )
    generate_and_save_tokens(
        de_tokenizer, german_texts, "dataset/german_tokens.txt", 12000
    )
