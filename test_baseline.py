import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models
from dataset.imagenet_mini import ImageNetMiniDataset
from utils import accuracy
from tqdm import tqdm
import csv
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='.', help='Root dir of images & data')
    parser.add_argument('--test_txt', default='data/test.txt')
    parser.add_argument('--checkpoint', required=True, help='Path to saved .pth')
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet18(pretrained=False, num_classes=100)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device)
    model.eval()

    print("🚀 開始測試 Baseline 模型 (RGB only)")
    test_ds = ImageNetMiniDataset(args.data_root, args.test_txt, transform, channels='RGB')
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    total_acc = 0.0
    with torch.no_grad():
        for imgs, labels, _ in tqdm(test_loader, desc="Testing RGB"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            total_acc += accuracy(outputs, labels).item()

    final_acc = total_acc / len(test_loader)
    print(f"📊 Test Accuracy (RGB): {final_acc:.4f}")

    # 儲存結果
    os.makedirs("logs_baseline", exist_ok=True)
    with open("logs_baseline/test_results.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["channels", "accuracy"])
        writer.writerow(["RGB", final_acc])
    print("✅ 測試結果已儲存至 logs_baseline/test_results.csv")
