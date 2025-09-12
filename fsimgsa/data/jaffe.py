from pathlib import Path
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image
import torchvision.transforms.functional as TF
import os


def get_jaffe_transform(transform_config):
    return v2.Compose([
        v2.Resize(transform_config.img_size),
        v2.Grayscale(num_output_channels=transform_config.output_channels)
    ])


class JaffeDataset(Dataset):

    def __init__(self, data_path: str, split: str, transform: v2.Transform):
        super(JaffeDataset, self).__init__()
        if not os.path.isdir(data_path):
            raise ValueError(f"{data_path} doesn't exist")

        self.rootdir = Path(os.path.join(data_path, split))
        self.files: list[tuple[Path, int]] = []
        self.classes: list[str] = []
        class_to_idx = {}
        self.transform = transform

        for i, class_dir in enumerate(sorted(self.rootdir.iterdir())):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            self.classes.append(class_name)
            class_to_idx[class_name] = i
            for file in class_dir.glob("*.tiff"):
                self.files.append((file, class_to_idx[class_name]))

        self.num_classes = len(self.classes)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file, class_ = self.files[index]
        img = Image.open(file)
        
        img = img.convert("L")
        img = TF.to_tensor(img)
        
        if self.transform is not None:
            img = self.transform(img)

        return img, class_
