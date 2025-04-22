# imagenet_dynamic_conv
Design a special convolutional module that is spatial size invariant and can handle an arbitrary number of input channels. 
# Dynamic Convolution for Variable Input Channels 

This repository contains the implementation for a dynamic convolution module that supports arbitrary input channel combinations (RGB, RG, R, etc.) based on the ImageNet-mini dataset.

## Folder Structure

- `models/`: contains the modified ResNet18 with dynamic channel support
- `dataset/`: includes the custom dataset loader with channel masking
- `logs/`: training logs and test results for the dynamic model
- `logs_baseline/`: training logs and test results for baseline ResNet18
- `figures/`: visualization plots for loss/accuracy comparison
- `train.py`, `test.py`: training and evaluation scripts
- `report.md`: final project report

## How to Run

```bash
# Training dynamic model
python train.py 

# Testing dynamic model
python test.py --checkpoint logs/your_model.pth

#Train baseline model
python train_baseline.py

# Testing baseline model
python test_baseline.py --checkpoint logs_baseline/baseline_model.pth

---

## 🧠 Reference
> Dynamic Convolution: Attention over Convolution Kernels, Wu et al., CVPR 2020
