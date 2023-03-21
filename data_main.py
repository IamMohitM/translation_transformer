# import torch
# import spacy
# from transformer.data_loader import make_eng_german_dataloader
# from transformer.data import load_dataset, generate_and_save_tokens
# from transformer.vocab import Vocab
# from transformer import de_tokenizer, en_tokenizer, german_tokens, english_tokens
import torch
from transformer.prepare_data import prepare_dataloader

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# torch.manual_seed(32)


# def generate_save_tokens(split_type, save_path_en, save_path_de, most_common=12000):
#     dataset = load_dataset("Multi30k", split_type, language_pair=["de", "en"])
#     de_tokenizer = spacy.load("de_core_news_sm")
#     en_tokenizer = spacy.load("en_core_web_sm")

#     german_texts = []
#     english_texts = []

#     for de_text, en_text in dataset:
#         german_texts.append(de_text)
#         english_texts.append(en_text)

#     generate_and_save_tokens(en_tokenizer, english_texts, save_path_en, most_common)
#     generate_and_save_tokens(de_tokenizer, german_texts, save_path_de, most_common)


if __name__ == "__main__":
    # german_tokens_path = "dataset/german_tokens.txt"
    # english_tokens_path = "dataset/english_tokens.txt"

    # dataloader = make_eng_german_dataloader(
    #     english_tokens_path, german_tokens_path, 32, device
    # )
    # # dataloader = make_eng_german_dataloader(
    # #     dataset, source_vocab, target_vocab, 32, device
    # # )

    # for data in dataloader:
    #     print(data)
    #     break

    # split_type = "valid"

    # generate_save_tokens(
    #     split_type,
    #     save_path_en=f"dataset/${split_type}_english_tokens.txt",
    #     save_path_de=f"dataset/${split_type}_german_tokens.txt",
    # )

    # split_type = "test"
    # generate_save_tokens(
    #     split_type,
    #     save_path_en=f"dataset/{split_type}_english_tokens.txt",
    #     save_path_de=f"dataset/{split_type}_german_tokens.txt",
    # )
    ...


    train_dataloader, source_vocab, target_vocab, _, _ = prepare_dataloader(2, 'train')

    for batch_num, batch_sample in enumerate(train_dataloader):
        
        source, target, labels, source_mask, target_mask = batch_sample
        print(source.shape)
        meh = source.clone()
