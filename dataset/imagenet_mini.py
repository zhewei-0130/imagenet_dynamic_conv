import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import random

class ImageNetMiniDataset(Dataset):
    def __init__(self, root_dir, txt_file, transform=None, channels="RGB"):
        self.root_dir = root_dir
        self.transform = transform
        self.fixed_channels = channels
        self.samples = []
        with open(txt_file, 'r') as f:
            for line in f:
                path, label = line.strip().split()
                self.samples.append((path, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        full_path = os.path.join(self.root_dir, path)
        img = Image.open(full_path).convert("RGB")

        # 🎲 控制通道組合機率
        prob = random.random()
        if prob < 0.10:
            selected_channels = 'RGB'
        elif prob < 0.40:
            selected_channels = random.choice(['RG', 'GB'])
        else:
            selected_channels = random.choice(['R', 'G', 'B'])

        r, g, b = img.split()
        channel_map = {'R': r, 'G': g, 'B': b}
        selected = [channel_map[c] for c in selected_channels]
        while len(selected) < 3:
            selected.append(Image.new("L", img.size))
        img = Image.merge("RGB", selected)

        if self.transform:
            img = self.transform(img)

        mask = torch.tensor([1 if c in selected_channels else 0 for c in 'RGB'], dtype=torch.float)
        return img, label, mask