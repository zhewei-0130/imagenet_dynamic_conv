import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset.imagenet_mini import ImageNetMiniDataset
from models.dynamic_resnet import ResNet18Dynamic
from utils import set_seed, save_checkpoint, accuracy, log_to_csv
from tqdm import tqdm

def evaluate(loader, model, device, criterion):
    model.eval()
    total_acc = 0.0
    total_loss = 0.0
    with torch.no_grad():
        for imgs, labels, masks in tqdm(loader, desc="Evaluating"):
            imgs, labels, masks = imgs.to(device), labels.to(device), masks.to(device)
            outputs = model(imgs, masks)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_acc += accuracy(outputs, labels).item()
    return total_acc / len(loader), total_loss / len(loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='.', help='Root directory of images & data')
    parser.add_argument('--train_txt', default='data/train.txt')
    parser.add_argument('--val_txt', default='data/val.txt')
    parser.add_argument('--channels', default='RGB', choices=['RGB','RG','GB','R','G','B'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    print("📁 準備資料集與轉換...")
    set_seed(42)
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
    train_ds = ImageNetMiniDataset(args.data_root, args.train_txt, transform, args.channels)
    val_ds   = ImageNetMiniDataset(args.data_root, args.val_txt, transform, args.channels)
    print("✅ 資料集載入完成")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 使用裝置: {device}")
    model = ResNet18Dynamic(num_classes=100, max_in_channels=3).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📦 模型參數總數: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, args.epochs+1):
        print(f"\n📦 Epoch {epoch}/{args.epochs}")
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        for imgs, labels, masks in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            imgs, labels, masks = imgs.to(device), labels.to(device), masks.to(device)
            outputs = model(imgs, masks)
            loss = criterion(outputs, labels)
            acc = accuracy(outputs, labels)
            total_loss += loss.item()
            total_acc += acc.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_loader)
        avg_train_acc = total_acc / len(train_loader)
        val_acc, val_loss = evaluate(val_loader, model, device, criterion)

        print(f"📊 Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f}")
        print(f"✅ Epoch {epoch} 完成 | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        log_to_csv("logs/train_log.csv", [epoch, avg_train_loss, avg_train_acc, val_loss, val_acc])

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, epoch, val_acc, f'checkpoint_epoch{epoch}.pth')