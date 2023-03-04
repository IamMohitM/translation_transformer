import torch
import spacy
from transformer.data_loader import make_eng_german_dataloader
from transformer.data import load_dataset
from transformer.vocab import Vocab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(32)

if __name__ == "__main__":
    german_tokens_path = "dataset/german_tokens.txt"
    english_tokens_path = "dataset/english_tokens.txt"
    dataloader = make_eng_german_dataloader(
        english_tokens_path, german_tokens_path, 32, device
    )
    # dataloader = make_eng_german_dataloader(
    #     dataset, source_vocab, target_vocab, 32, device
    # )

    for data in dataloader:
        print(data)
        break
