import torch
import spacy

UNK_IDX = 0
PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

special_tokens = {
    UNK_IDX: "<unk>",
    PAD_IDX: "<pad>",
    SOS_IDX: "<sos>",
    EOS_IDX: "<eos>",
}

english_token_path = "dataset/english_tokens.txt"
german_token_path = "dataset/german_tokens.txt"

de_tokenizer = spacy.load("de_core_news_sm")
en_tokenizer = spacy.load("en_core_web_sm")


with open(english_token_path, "r") as f:
    german_tokens = f.read().strip().split("\n")

with open(german_token_path, "r") as f:
    english_tokens = f.read().strip().split("\n")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")