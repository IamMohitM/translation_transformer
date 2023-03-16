import torch
from .data_conf import de_tokenizer, en_tokenizer, english_tokens, german_tokens
from .vocab import Vocab, EOS_IDX, SOS_IDX

from .transformer import EncoderDecoder

from .prepare_data import build_vocab, generate_subsequent_mask, load_tokenizer, get_transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def greedy_decode(model: EncoderDecoder, token_indices,max_len, start_symbol=SOS_IDX):

    # source_mask = torch.ones((1, 1, len(token_indices)), dtype = torch.bool, device=device)
    source_mask = None
    token_indices = token_indices.to(device).unsqueeze(0)

    memory = model.encoder(token_indices, source_mask)
    ys = (
        torch.ones(len(token_indices), len(token_indices))
        .fill_(start_symbol)
        .type(torch.long)
        .to(device)
    )
    for i in range(max_len - 1):
        target_mask = generate_subsequent_mask(ys.size(0)).type(torch.bool).to(device)
        decoder_state = model.decoder.init_state(memory, source_mask)
        out = model.decoder(ys, decoder_state, target_mask)[0]
        # prob = 
        next_word = torch.argmax(out, dim = -1).item()

        if next_word == EOS_IDX:
            break

        ys = torch.cat([ys, torch.ondex(1,1).type_as(token_indices).fill_(next_word)], dim = 0)

    return ys

def load_vocab(path: str):
    return torch.load(path)


def translate(model, source_vocab, source_tokenizer, target_vocab, target_tokenizer, german_text: str):
    model.eval()
    transforms = get_transforms(source_tokenizer, source_vocab)

    token_indices = transforms(german_text.rstrip("\n"))

    num_tokens = token_indices.shape[0]

    # source_mask = torch.ones(num_tokens, num_tokens)

    tgt_tokens = greedy_decode(
        model, token_indices, max_len=num_tokens + 5, start_symbol=SOS_IDX
    )

    return " ".join(target_vocab.lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<sos>", "").replace("<eos>", "")

    return tgt_tokens
