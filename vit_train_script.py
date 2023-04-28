from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
import wandb
from transformer.imagenette_dataset import Imagenette

torch.manual_seed(50)

from transformer.vision_transformer import ViT, ViTSequential

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        checkpoint_path="checkpoints/model_imagenette.pth",
    ) -> None:
        self.optimizer = optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.val_acc = float("-inf")
        self.checkpoint_path = checkpoint_path

    def predict(self, model, *args):
        return model(*args)

    def train(
        self,
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        loss_fn,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        scheduler: torch.optim.lr_scheduler = None,
    ):
        wandb.watch(model, loss_fn, log="all", log_freq=1000)
        model.to(self.device)
        for epoch_num in range(epochs):
            print(f"Epoch {epoch_num}")
            for batch_input, batch_labels in train_dataloader:
                batch_input, batch_labels = batch_input.to(
                    self.device
                ), batch_labels.to(self.device)
                loss_value = self.training_step(
                    model, loss_fn, batch_labels, batch_input
                )

                wandb.log({"training_loss": loss_value})
            self.evaluate(model, val_dataloader)
            if scheduler:
                scheduler.step(self.val_acc)
                wandb.log({"learning_rate": self.optimizer.param_groups[-1]["lr"]})

    def training_step(self, model, loss_fn, labels, *args):
        output = self.predict(model, *args)
        loss = loss_fn(output, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()

    def save_model(self, model: torch.nn.Module):
        print(f"Saving model to {self.checkpoint_path}")
        torch.save({"state_dict": model.state_dict()}, self.checkpoint_path)

    def evaluate(self, model, dataloader):
        model.eval()
        correct = 0
        total = 0
        labels_list = []
        predictions_list = []
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            labels_list.append(labels)

            # test = Variable(images.view(100, 1, 28, 28))

            outputs = model(images)

            predictions = torch.max(outputs, 1)[1].to(device)
            predictions_list.append(predictions)
            correct += (predictions == labels).sum()

            total += len(labels)

        accuracy = correct / total
        print(f"Val Accuracy: {accuracy}")
        wandb.log({"val_accuracy": accuracy})
        if accuracy > self.val_acc:
            # wandb.log_artifact(model)
            self.val_acc = accuracy
            self.save_model(model)
            torch.onnx.export(model, images, "model.onnx")
            wandb.save("model.onnx")


img_size = (224, 224)
batch_size = 32
train_augment = transforms.Compose(
    [
        transforms.Resize(img_size),
        transforms.ColorJitter(0.2, 0.2, 0.2),
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

if __name__ == "__main__":
    config = {
        "patch_size": 16,
        "embed_dim": 512,
        "mlp_num_hidden": 1024,
        "num_heads": 32,
        "num_blocks": 8,
        "emb_dropout": 0.4,
        "block_dropout": 0.4,
        "qkv_transform_dim": 2048,
        "use_bias": True,
        "img_size": img_size,
        "image_channels": 3,
        "dataset_name": "Imagenette",
        "lr": 0.0001,
        "optimizer": "Adam",
        "epochs": 100,
        "schduler_plateu": {
            "mode": "max",
            "factor": 0.9,
            "patience": 2,
            "min_lr": 0.00001,
        },
        "batch_size": batch_size,
        "Notes": "Added color jitter",
    }

    vit = ViT(**config)

    wandb.init(project="Imagenette_ViT", config=config)

    train_dataset = Imagenette(
        file_path="/home/mohitm/datasets/imagenette2/train_file.txt",
        transforms=train_augment,
    )
    val_dataset = Imagenette(
        file_path="/home/mohitm/datasets/imagenette2/val_file.txt",
        transforms=val_augment,
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=4, shuffle=True
    )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        vit.parameters(),
        lr=config.get("lr", 0.0005),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, verbose=True, **config["schduler_plateu"]
    )
    trainer = Trainer(optimizer=optimizer)
    trainer.train(
        vit,
        train_dataloader,
        val_dataloader,
        loss_fn,
        optimizer,
        scheduler=scheduler,
        epochs=config.get("epochs", 100),
    )
