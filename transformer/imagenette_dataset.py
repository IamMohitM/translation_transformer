import torch
from PIL import Image


class Imagenette(torch.utils.data.Dataset):
    def __init__(self, file_path: str, transforms) -> None:
        super().__init__()
        self.tranforms = transforms
        with open(file_path, "r") as f:
            data = f.read().strip().split("\n")
        self.image_paths, self.labels = zip(*[val.split(" ") for val in data])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image = Image.open(self.image_paths[index]).convert("RGB")
        label = int(self.labels[index])
        image = self.tranforms(image)
        return image, torch.tensor(label)
