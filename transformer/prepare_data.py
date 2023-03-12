import torch
import torchtext
from torchtext.datasets import multi30k, Multi30k
from .data_conf import UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX, special_tokens, device
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import DataLoader

from typing import List

# We need to modify the URLs for the dataset since the links to the original dataset are broken
# Refer to https://github.com/pytorch/text/issues/1756#issuecomment-1163664163 for more info
multi30k.URL[
    "train"
] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL[
    "valid"
] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"

# token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
# token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')

source_language = "de"
target_language = "en"
language_pair = (source_language, target_language)
language_index = {source_language: 0, target_language: 1}


def load_tokenizer(language_key: str):
    return torchtext.data.get_tokenizer("spacy", language_key)


def get_dataset_iter(split_type=["en"], language_pair=language_pair):
    return Multi30k(split=split_type, language_pair=language_pair)


def build_vocab(tokenizer, data_iter, language):
    # tokenizer = load_tokenizer(language)

    def data_iterator(data_iter, language):
        for sample in data_iter:
            yield tokenizer(sample[language_index[language]])

    vocab = build_vocab_from_iterator(
        data_iterator(data_iter, language),
        specials=special_tokens.values(),
        special_first=True,
        min_freq=1,
    )

    vocab.set_default_index(UNK_IDX)

    return vocab


def generate_subsequent_mask(size):
    return torch.tril(torch.full((size, size), 1))


def create_mask(source_sequence, target_sequence):
    source_length = source_sequence.shape[0]
    target_length = target_sequence.shape[0]
    assert source_length == target_length

    source_pad_mask = (source_sequence != PAD_IDX).unsqueeze(1)

    target_pad_mask = (target_sequence != PAD_IDX).unsqueeze(1)

    max_target_sequence_length = target_sequence.shape[1]
    # subsequent mask
    target_subsequence_mask = generate_subsequent_mask(
        max_target_sequence_length
    ).unsqueeze(0)

    return source_pad_mask, target_pad_mask & target_subsequence_mask

    # source_mask = torch.zeros((source_length, source_length), device=device)

    # # where source sequence is padded
    # source_padding_mask = source_sequence == PAD_IDX

    # # where target sequence is padded
    # target_padding_mask = target_sequence == PAD_IDX

    # return (
    #     source_mask,
    #     target_subsequence_mask,
    #     source_padding_mask,
    #     target_padding_mask,
    # )


def sequential_transformation(*transforms):
    def func(text):
        for transform in transforms:
            text = transform(text)
        return text

    return func


def prepare_input_tokens(token_indexes: List[int]):
    return torch.cat(
        [torch.tensor([SOS_IDX]), torch.tensor(token_indexes), torch.tensor([EOS_IDX])]
    ).long()


def get_transforms(tokenizer, vocab):
    return sequential_transformation(tokenizer, vocab, prepare_input_tokens)


def prepare_dataloader(batch_size, split_type, source_vocab=None, target_vocab=None):
    source_tokenizer = load_tokenizer(source_language)
    target_tokenizer = load_tokenizer(target_language)

    dataset = get_dataset_iter(split_type, language_pair)
    if source_vocab is None or target_vocab is None:
        source_vocab = build_vocab(source_tokenizer, dataset, source_language)
        target_vocab = build_vocab(target_tokenizer, dataset, target_language)

    source_transform = get_transforms(source_tokenizer, source_vocab)
    target_transform = get_transforms(target_tokenizer, target_vocab)

    def collate_fn(batch):
        source_batch, target_batch = [], []
        for source_sample, target_sample in batch:
            source_batch.append(source_transform(source_sample.rstrip("\n")))
            target_batch.append(target_transform(target_sample.rstrip("\n")))

        source_batch = pad_sequence(
            source_batch, padding_value=PAD_IDX, batch_first=True
        )
        target_batch = pad_sequence(
            target_batch, padding_value=PAD_IDX, batch_first=True
        )

        label_batch = target_batch[:, 1:]  # No SOS token
        target_batch = target_batch[:, :-1]

        source_mask, target_mask = create_mask(source_batch, target_batch)

        all_batches = [
            source_batch,
            target_batch,
            label_batch,
            source_mask,
            target_mask,
        ]

        # move everything to the target device
        return [x.to(device) for x in all_batches]

    return (
        DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn),
        source_vocab,
        target_vocab,
    )
