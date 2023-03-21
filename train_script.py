import torch

from transformer.transformer import EncoderDecoder
from transformer.data_loader import PAD_IDX
from transformer.optimizer import NoamOpt
from torch.utils.tensorboard import SummaryWriter

from transformer.prepare_data import prepare_dataloader

writer = SummaryWriter(log_dir="tensorboard_logs")

from typing import Tuple

##param definition
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# num_heads = 8
# embed_dim = 128
# batch_size = 128
# ffn_num_hidden = 2048
# total_words = 4
# num_hidden = 64  # dq = dk = dv = num_hiddens
# num_blocks = 6
# dropout = 0.1
# vocab_size = 12004
# warmup = 1000

num_epochs = 10
min_val_loss = float("inf")

german_tokens_path = "dataset/german_tokens.txt"
english_tokens_path = "dataset/english_tokens.txt"
valid_english_tokens_path = "dataset/german_tokens"

checkpoint_path = "checkpoints/model.pt"
# model dev
global_steps = 0


def get_model(
    embed_dim,
    num_hidden,
    ffn_num_hidden,
    num_heads,
    num_blocks,
    source_vocab_size,
    target_vocab_size,
):
    return EncoderDecoder(
        embed_dim=embed_dim,
        num_hidden=num_hidden,
        ffn_hidden=ffn_num_hidden,
        num_heads=num_heads,
        num_blocks=num_blocks,
        source_vocab_size=source_vocab_size,
        target_vocab_size=target_vocab_size,
    ).to(device)


def translate_batch(source, truth, target):
    preds = torch.argmax(target, dim=-1).tolist()
    source_tokens = source_vocab.lookup_tokens(source.tolist())
    truth_tokens = target_vocab.lookup_tokens(truth.tolist())
    target_tokens = target_vocab.lookup_tokens(preds)
    return source_tokens, truth_tokens, target_tokens


def predict(model, data):
    source_batch, target_batch, label_batch, source_mask, target_mask = data
    return model(source_batch, target_batch, source_mask, target_mask), label_batch


def train_step(
    model: torch.nn.Module,
    data: Tuple,
    loss_fn: torch.nn.CrossEntropyLoss,
    optimizer: NoamOpt,
    step: int,
):
    global global_steps
    print(f"Epoch Step - {step}")
    optimizer.zero_grad()
    output, label_batch = predict(model, data)
    # print(translate_batch(data[0][0], output[0]))
    loss = loss_fn(output.reshape(-1, output.shape[-1]), label_batch.reshape(-1))
    loss.backward()
    optimizer.step()
    global_steps += 1
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
    global global_steps
    for batch_num, batch_sample in enumerate(dataloader):
        loss = train_step(model, batch_sample, loss_fn, optimizer, batch_num)
        writer.add_scalar("Train Step Loss", loss, global_steps)
        losses.append(loss)

    print(f"Epoch {epoch_num} mean loss - {torch.tensor(losses).mean(): .2f}")
    return losses


def evaluate(model, dataloader):
    global min_val_loss
    losses = []
    with torch.no_grad():
        for data in dataloader:
            output, label_batch = predict(model, data)
            loss = loss_fn(
                output.reshape(-1, output.shape[-1]), label_batch.reshape(-1)
            )
            losses.append(loss)
    print(f"Evaluation Mean loss {(mean_loss:=torch.tensor(losses).mean())}")

    return mean_loss


def save_model(model, optimizer, epoch, val_loss, path=checkpoint_path):
    global min_val_loss
    if val_loss < min_val_loss:
        print(f"Min Val loss at epoch {epoch}. Saving Model.")
        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.optimizer.state_dict(),
                "loss": val_loss,
            },
            checkpoint_path,
        )
        min_val_loss = val_loss


def train(
    model: torch.nn.Module,
    train_dataloader,
    validation_dataloader,
    optimizer: NoamOpt,
    loss_fn,
    num_epochs: KeyboardInterrupt,
):
    training_losses = []
    validation_losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = train_epoch(model, train_dataloader, optimizer, loss_fn, epoch)
        mean_epoch_loss = torch.tensor(epoch_losses).mean()
        writer.add_scalar("Epoch Mean Loss", mean_epoch_loss, epoch)
        training_losses.append(mean_epoch_loss)

        model.eval()
        mean_val_loss = evaluate(model, validation_dataloader)
        writer.add_scalar("Validation Mean Loss", mean_val_loss, epoch)
        validation_losses.append(mean_val_loss)

        writer.flush()

        save_model(model, optimizer, epoch, mean_val_loss)

    return training_losses, validation_losses


if __name__ == "__main__":
    # params
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_heads = 8

    embed_dim = 512
    batch_size = 128
    # decreasgin ffn_num_hiden to decrease overfitting possibilities
    ffn_num_hidden = 512
    num_hidden = 64  # dq = dk = dv = num_hiddens
    # decreasing num_blocks
    num_blocks = 6
    dropout = 0.1
    num_epochs = 100

    (
        train_dataloader,
        source_vocab,
        target_vocab,
        source_tokenizer,
        target_tokenizer,
    ) = prepare_dataloader(batch_size=batch_size, split_type="train")

    val_dataloader, _, _, _, _ = prepare_dataloader(
        batch_size=batch_size,
        split_type="valid",
        source_vocab=source_vocab,
        target_vocab=target_vocab,
    )

    # test_dataloader, _, _ = prepare_dataloader(batch_size=batch_size, split_type='test', source_vocab=source_vocab, target_vocab=target_vocab)

    source_vocab_size = len(source_vocab)
    target_vocab_size = len(target_vocab)

    model = get_model(
        embed_dim,
        num_hidden,
        ffn_num_hidden,
        num_heads,
        num_blocks,
        source_vocab_size,
        target_vocab_size,
    )

    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    adam = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    optimizer = NoamOpt(embed_dim=embed_dim, optimizer=adam)
    # ## Data definition
    # X = torch.randint(low=0, high=vocab_size, size=(3, total_words), device=device)
    train_dataloader, source_vocab, target_vocab, _, _ = prepare_dataloader(
        batch_size=batch_size, split_type="train"
    )
    val_dataloader, source_vocab, target_vocab, _, _ = prepare_dataloader(
        batch_size=batch_size,
        split_type="valid",
        source_vocab=source_vocab,
        target_vocab=target_vocab,
    )

    losses = []

    training_losses, validation_losses = train(
        model, train_dataloader, val_dataloader, optimizer, loss_fn, num_epochs
    )

    writer.close()
