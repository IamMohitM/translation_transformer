"""
    Credits: https://kikaben.com/transformers-data-loader/
"""
import spacy
from typing import List

# special token indices
UNK_IDX = 0
PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

UNK = "<unk>"  # Unknown
PAD = "<pad>"  # Padding
SOS = "<sos>"  # Start of sentence
EOS = "<eos>"  # End of sentence

SPECIAL_TOKENS = [UNK, PAD, SOS, EOS]


class Vocab:
    def __init__(
        self, tokenizer: spacy.language.Language, tokens: List[str] = []
    ) -> None:
        self.tokenizer = tokenizer
        self.tokens = SPECIAL_TOKENS + tokens
        self.index_lookup = {self.tokens[i]: i for i in range(len(self.tokens))}

    def __len__(self) -> int:
        return len(self.tokens)  # vocab size

    def __call__(self, text: str) -> List[int]:
        text = text.strip()
        return [self.to_index(token.text) for token in self.tokenizer(text)]

    def to_index(self, token: str) -> int:
        return self.index_lookup[token] if token in self.index_lookup else UNK_IDX


if __name__ == "__main__":
    english_tokenizer = spacy.load("en_core_web_sm")
    with open("dataset/english_tokens.txt", "r") as f:
        tokens = f.read().strip().split("\n")
    vocab = Vocab(tokenizer=english_tokenizer, tokens=tokens)
    indices = vocab("hello, world!")
    print(indices)
