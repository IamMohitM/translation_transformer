import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from transformer.vision_transformer import ViT

img_size = (56, 56)

val_augment = transforms.Compose(
    [
        transforms.Resize(img_size),
        transforms.PILToTensor(),
        lambda x: x.float(),
        lambda x: x / 255.0,
    ]
)

config = {
    "patch_size": 8,
    "embed_dim": 512,
    "mlp_num_hidden": 1024,
    "num_heads": 16,
    "num_blocks": 8,
    "emb_dropout": 0.1,
    "block_dropout": 0.1,
    "use_bias": True,
    "img_size": img_size,
    "image_channels": 1,
}

checkpoint_path = "checkpoints/vit_model.pth"
model = ViT(**config)

model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
model.eval()
model = model.cuda()


def preprocess_image(image):
    """Simply the function to preprocess the image"""

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = Image.fromarray(image)
    image = val_augment(image)
    image = image.unsqueeze(0)
    image = image.cuda()
    return image


def predict(image):
    image = preprocess_image(image)
    preds = model(image)
    probs = torch.nn.functional.softmax(preds, dim=1)
    return probs



if __name__ == "__main__":
    ...
