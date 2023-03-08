import torch
from d2l import torch as d2l
from transformer.encoder import Encoder, EncoderBlock
from transformer.decoder import DecoderBlock, Decoder
from transformer.transformer import EncoderDecoder
from transformer.data_loader import make_eng_german_dataloader, PAD_IDX
from transformer.optimizer import NoamOpt

from typing import Tuple


def train_step(
    model: torch.nn.Module,
    data: Tuple,
    loss_fn: torch.nn.CrossEntropyLoss,
    optimizer: NoamOpt,
    step: int
):
    print(f"Epoch Step - {step}")
    source_batch, target_batch, label_batch, source_mask, target_mask = data
    output = model(source_batch, target_batch, source_mask, target_mask)
    loss = loss_fn(output.reshape(-1, output.shape[-1]), label_batch.reshape(-1))
    loss.backward()
    optimizer.step()
    return loss.detach().cpu().item()


def train_epoch(
    model: torch.nn.Module,
    dataloader,
    optimizer: NoamOpt,
    loss_fn: torch.nn.CrossEntropyLoss,
    epoch_num: int,
) -> torch.tensor:
    print(f"Epoch {epoch_num}")
    losses = []

    for step, data in enumerate(dataloader):
        loss = train_step(model, data, loss_fn, optimizer, step)
        losses.append(loss)

    print(f"Epoch {epoch_num} mean loss - {torch.mean(losses)}")
    return losses


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_heads = 8
    embed_dim = 128
    batch_size = 128
    ffn_num_hidden = 256
    total_words = 4
    num_hidden = 64  # dq = dk = dv = num_hiddens
    num_blocks = 2
    dropout = 0.1
    vocab_size = 12005
    num_epochs = 500

    # Encoder ----
    # encoder = Encoder(
    #     num_hidden, num_heads, embed_dim, ffn_hidden, num_blocks=2, vocab_size=200
    # )

    # X = torch.ones((2, total_words, embed_dim))
    # encoder_output = encoder(X)
    # print(encoder_output.shape)
    ## ------

    # Decoder ------
    # encoder_block = EncoderBlock(num_heads, num_hidden, embed_dim, ffn_num_hidden)
    # valid_lens = torch.tensor([5, 10])
    # decoder_blk = DecoderBlock(num_heads, num_hidden, embed_dim, ffn_num_hidden, 0)
    # X = torch.ones((2, total_words, embed_dim))
    # encoder_block_output = encoder_block(X, valid_lens)
    # state = [encoder_block_output, valid_lens, [None]]
    # decoder_block_ouput = decoder_blk(X, state)
    # print(d2l.check_shape(decoder_block_ouput[0], X.shape))
    # print(encoder_)
    ## -------

    ## Transformer --------

    # encoder = Encoder(
    #     vocab_size=len(data.src_vocab),
    #     num_hidden=num_hidden,
    #     ffn_hidden=ffn_num_hidden,
    #     num_heads=num_heads,
    #     num_blocks=num_blocks,
    #     dropout=dropout,
    #     embed_dim=embed_dim,
    # )
    # decoder = Decoder(
    #     vocab_size=len(data.tgt_vocab),
    #     num_hidden=num_hidden,
    #     ffn_hidden=ffn_num_hidden,
    #     num_heads=num_heads,
    #     num_blocks=num_blocks,
    #     dropout=dropout,
    #     embed_dim=embed_dim,
    # )
    # model = d2l.Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab["<pad>"], lr=0.0015)
    # trainer = d2l.Trainer(max_epochs=30, gradient_clip_val=0, num_gpus=1)
    # trainer.fit(model, data)
    # ----------
    ##-EncoderDecoder

    model = EncoderDecoder(
        embed_dim=embed_dim,
        num_hidden=num_hidden,
        ffn_hidden=ffn_num_hidden,
        num_heads=num_heads,
        num_blocks=num_blocks,
        vocab_size=vocab_size,
    )

    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    adam = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    optimizer = NoamOpt(embed_dim=embed_dim, optimizer=adam)
    # ## Data definition
    # X = torch.randint(low=0, high=vocab_size, size=(3, total_words), device=device)
    german_tokens_path = "dataset/german_tokens.txt"
    english_tokens_path = "dataset/english_tokens.txt"
    dataloader = make_eng_german_dataloader(
        english_tokens_path, german_tokens_path, batch_size, device
    )
    # dataloader = make_eng_german_dataloader(
    #     dataset, source_vocab, target_vocab, 32, device
    # )
    losses = []

    for epoch in range(num_epochs):
        epoch_loss = train_epoch(model, dataloader, optimizer, loss_fn, epoch)
        losses.append(epoch_loss)

    # for data in dataloader:
    #     source_batch, target_batch, label_batch, source_mask, target_mask = data

    #     output = model(source_batch, target_batch, source_mask, target_mask)
    #     optimizer.step()
    #     loss = loss_fn(output.reshape(-1, output.shape[-1]), label_batch.reshape(-1))
    #     loss.backward()
    #     losses.append(loss.detach().cpu().item())

    # print(f"Mean - {torch.mean(losses)}")

    # output = model(source_batch, target_batch)
    # print(output)
