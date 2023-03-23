from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch
from torch.utils.data import DataLoader

from transformer.vision_transformer import ViT, ViTSequential

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer
        pass

    def predict(self, model, *args):
        return model(*args)
        ...

    def train(
        self, model, train_dataloader, val_dataloader, loss_fn, optimizer, epochs
    ):
        # for batch_input, batch_labels in train_dataloader:
        # loss_value =

        ...

    def training_step(self, model, loss_fn, labels, *args):
        output = self.predict(model, *args)
        labels = ...
        loss = loss_fn(output, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()
        ...

    def evaluate():
        ...


img_size = 56
batch_size = 128
train_augment = transforms.Compose(
    [
        transforms.Resize(img_size),
        transforms.RandomRotation(60),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.PILToTensor(),
        lambda x: x.float(),
        lambda x: x / 255.0,
    ]
)

val_augment = transforms.Compose(
    [
        transforms.Resize(img_size),
        transforms.PILToTensor(),
        lambda x: x.float(),
        lambda x: x / 255.0,
    ]
)

train_dataset = FashionMNIST(
    root="dataset/fmnist", train=True, download=False, transform=train_augment
)
val_dataset = FashionMNIST(
    root="dataset/fmnist", train=False, download=False, transform=val_augment
)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)


def train(model, train_dataloader, val_dataloader, loss_fn, optimizer, num_epochs=50):
    count = 0
    # Lists for visualization of loss and accuracy
    loss_list = []
    iteration_list = []
    accuracy_list = []

    # Lists for knowing classwise accuracy
    predictions_list = []
    labels_list = []
    model.to(device)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}")
        model.train()
        for images, labels in train_dataloader:
            # Transfering images and labels to GPU if available
            images, labels = images.to(device), labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            # Initializing a gradient as 0 so there is no mixing of gradient among the batches
            optimizer.zero_grad()
            # Propagating the error backward
            loss.backward()
            # Optimizing the parameters
            optimizer.step()
            count += 1

        model.eval()
        correct = 0
        total = 0
        for images, labels in val_dataloader:
            images, labels = images.to(device), labels.to(device)
            labels_list.append(labels)

            # test = Variable(images.view(100, 1, 28, 28))

            outputs = model(images)

            predictions = torch.max(outputs, 1)[1].to(device)
            predictions_list.append(predictions)
            correct += (predictions == labels).sum()

            total += len(labels)

        accuracy = correct / total
        loss_list.append(loss.data)
        iteration_list.append(count)
        accuracy_list.append(accuracy)

        # if not (count % 500):
        print(
            "Iteration: {}, Loss: {}, Accuracy: {}%".format(count, loss.data, accuracy)
        )


if __name__ == "__main__":
    patch_size = 8
    embed_dim = 512
    mlp_num_hidden = 1024
    num_heads = 16
    num_blocks = 8
    emb_dropout = 0.1
    block_dropout = 0.1
    use_bias = True


    # vit = ViT(
    #     img_size=img_size,
    #     patch_size=patch_size,
    #     embed_size=embed_dim,
    #     mlp_num_hiddens=mlp_num_hidden,
    #     num_heads=num_heads,
    #     num_blocks=num_blocks,
    #     emb_dropout=emb_dropout,
    #     block_dropout=block_dropout,
    # )

    vit = ViTSequential(
            img_size=img_size,
            patch_size=patch_size,
            embed_size=embed_dim,
            mlp_num_hiddens=mlp_num_hidden,
            num_heads=num_heads,
            num_blocks=num_blocks,
            emb_dropout=emb_dropout,
            block_dropout=block_dropout,
        )

    loss_fn = torch.nn.CrossEntropyLoss()
    train(
        vit,
        train_dataloader,
        val_dataloader,
        loss_fn,
        optimizer=torch.optim.Adam(vit.parameters(), lr=0.0001, ),
    )
