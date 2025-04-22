import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset.imagenet_mini import ImageNetMiniDataset
from models.dynamic_resnet import ResNet18Dynamic
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

    channel_variants = ['RGB', 'RG', 'GB', 'R', 'G', 'B']
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet18Dynamic(num_classes=100, max_in_channels=3)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device)
    model.eval()

    print("🚀 開始測試六種通道組合：")
    results = []
    for channels in channel_variants:
        print(f"\n📁 測試通道: {channels}")
        test_ds = ImageNetMiniDataset(args.data_root, args.test_txt, transform, channels)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

        total_acc = 0.0
        with torch.no_grad():
            for imgs, labels, masks in tqdm(test_loader, desc=f"Testing {channels}"):
                imgs, labels, masks = imgs.to(device), labels.to(device), masks.to(device)
                outputs = model(imgs, masks)
                total_acc += accuracy(outputs, labels).item()
        final_acc = total_acc / len(test_loader)
        print(f"📊 Test Accuracy ({channels}): {final_acc:.4f}")
        results.append((channels, final_acc))

    # 儲存結果
    os.makedirs("logs", exist_ok=True)
    with open("logs/test_results.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["channels", "accuracy"])
        writer.writerows(results)
    print("✅ 測試結果已儲存至 logs/test_results.csv")