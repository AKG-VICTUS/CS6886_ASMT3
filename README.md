# CS6886_SYS4DL_ASMT3
---

# 📦 MobileNetV2 Quantization on CIFAR-10

This repository provides a reproducible pipeline to:

1. Train a **baseline MobileNetV2** model on CIFAR-10.
2. Apply a **configurable custom quantization scheme** for weights & activations.
3. Report **accuracy, storage size, and compression ratios**.
4. Log results automatically using **Weights & Biases (W&B)**.

---

## 🚀 Features

* **Dataset Preparation**: CIFAR-10 with normalization & augmentation.
* **Model**: MobileNetV2, adapted to 10 classes.
* **Quantization**: Symmetric linear quantization with configurable bitwidths.
* **Metrics**: Accuracy (FP32 & quantized), model size in MB, compression ratios.
* **Reproducibility**: Fixed seeds, version-pinned dependencies.
* **Experiment Tracking**: W&B logging for all configs.

---

## 📂 Repository Structure

```
mobilenetv2_quant/
│
├── checkpoints/              
│   └── mobilenetv2_cifar10_fp32.pth   # Baseline FP32 model (generated after training)
│
├── data/                     # CIFAR-10 dataset (downloaded automatically)
│
├── dataloader.py             # Dataset loading, normalization, augmentation
├── mobilenet_v2_model.py     # MobileNet-V2 model definition (CIFAR-10 adapted)
├── train_mobilenetv2.py      # FP32 training script
├── quantize_improved.py      # Custom configurable quantization (weights/activations)
├── quantize_mobilenetv2_multi.py # Multiple quantization configs & W&B logging
├── utils.py                  # Evaluation, model size, compression metrics
└── README.md                 # Documentation & run instructions

```

---

## ⚙️ Environment Setup

Create and activate a virtual environment:

```bash
python3 -m venv mobilenetv2_quant
source mobilenetv2_quant/bin/activate
```

Install dependencies:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129  ##for GPU with CUDA12.9

## or

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu    ##for CPU

## or

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4  ##for AMD GPU

pip install wandb

pip install numpy tqdm

---
## wandb login

wandb login    ## enter API key upon prompting

## 📊 Running Experiments

### 1. Train & Evaluate Baseline + Quantized Models

```bash
python3 train_mobilenetv2.py

python3 quantize_mobilenetv2_multi.py
```

These scripts:

* Trains MobileNetV2 on CIFAR-10 (baseline FP32).
* Applies multiple quantization configs (e.g., 8-bit, 6-bit, 4-bit).
* Evaluates accuracy, computes model size, logs results.

---

## 📈 Example Results

Console output:

```
FP32 Test Accuracy: 86.5%

=== Quantization: Weights=8 | Activations=8 ===
Quantized Test Accuracy: 74.5%
FP32 model size:   8.53 MB
Quantized size:    2.18 MB
Compression ratio: 3.91x
```

W&B dashboard:

* FP32 vs Quantized accuracy comparison.
* Model size & compression ratio tracked as metrics.
* Parallel charts across multiple configs.

---

## 📌 Reproducibility

* **Dataset**: CIFAR-10 auto-downloaded with checksum.
* **Transforms**:

  * Random crop (32×32, padding=4)
  * Random horizontal flip
  * Normalization (mean/std of CIFAR-10)
* **Training Strategy**:

  * Optimizer: SGD (lr=0.1, momentum=0.9, weight decay=5e-4)
  * Scheduler: StepLR
  * Epochs: 300
* **Seeds**: Fixed (`torch`, `numpy`, `random`).
* **Environment**: Versions used: torch-2.8.0, torchvision-0.23.0, wandb, numpy, tqdm-4.67.1

---

## 📌 Compression Metrics

* **Weight Compression Ratio** = FP32 weight size ÷ Quantized weight size.
* **Activation Compression Ratio** = FP32 activation footprint ÷ Quantized activation footprint.
* Both are reported per config.

---

## 📜 License

This repo is for educational purposes under fair use.

---



