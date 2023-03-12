import torch
from .encoder import Encoder
from .decoder import Decoder


class EncoderDecoder(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_hidden: int,
        ffn_hidden: int,
        num_heads: int,
        num_blocks: int,
        source_vocab_size: int,
        target_vocab_size: int,
        dropout: float = 0.1,
        max_length: int = 1000,
    ) -> None:
        super().__init__()
        self.encoder: Encoder = Encoder(
            num_hidden,
            num_heads=num_heads,
            embed_dim=embed_dim,
            ffn_hidden=ffn_hidden,
            vocab_size=source_vocab_size,
            max_length=max_length,
            num_blocks=num_blocks,
            dropout=dropout,
        )
        self.decoder: Decoder = Decoder(
            vocab_size=target_vocab_size,
            num_heads=num_heads,
            num_hidden=num_hidden,
            embed_dim=embed_dim,
            ffn_hidden=ffn_hidden,
            num_blocks=num_blocks,
            max_length=max_length,
            dropout=dropout,
        )

    def forward(
        self,
        encoder_input,
        decoder_input,
        encoder_valid_lens=None,
        decoder_valid_lens=None,
    ):
        encoder_output = self.encoder(encoder_input, encoder_valid_lens)
        decoder_state = self.decoder.init_state(encoder_output, encoder_valid_lens)
        return self.decoder(decoder_input, decoder_state, decoder_valid_lens)[0]
