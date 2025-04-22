import torch
import random
import numpy as np
import os
import csv

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def accuracy(outputs, labels):
    _, preds = outputs.max(1)
    return (preds == labels).float().mean()

def save_checkpoint(model, epoch, acc, filename):
    state = {'epoch': epoch, 'state_dict': model.state_dict(), 'accuracy': acc}
    torch.save(state, filename)

def log_to_csv(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writerow(row)