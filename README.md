# ResNet From Scratch (PyTorch)

An educational implementation of Deep Residual Networks (ResNet) built completely from scratch in PyTorch without relying on pre-trained model weights.

---

## Table of Contents
- [Architecture Overview](#-architecture-overview)
- [Project Structure](#-project-structure)
- [Requirements & Installation](#-requirements--installation)
- [How to Run](#-how-to-run)
- [Expected Results & Metrics](#-expected-results--metrics)
- [References](#-references)

---

## Architecture Overview

### The Degradation Problem
As deep neural networks increase in depth, model accuracy often saturates and then degrades rapidly. Crucially, this is not caused by overfitting (as training error also increases), but rather by optimization difficulties: standard stacked layers struggle to learn identity mappings, causing backpropagated gradients to diminish across dozens of layers.

### Residual Learning Solution
ResNet (*He et al., 2015*) addresses this degradation problem by reformulating convolutional layers to fit a residual mapping $\mathcal{F}(\mathbf{x}) = \mathcal{H}(\mathbf{x}) - \mathbf{x}$ rather than directly fitting an unconstrained underlying mapping $\mathcal{H}(\mathbf{x})$.

The explicit mathematical formulation for a residual block output is:

$$\mathbf{y} = \text{ReLU}\left(\mathcal{F}(\mathbf{x}, \{W_i\}) + W_s \mathbf{x}\right)$$

where:
- $\mathbf{x}$ and $\mathbf{y}$ are input and output vectors of the block.
- $\mathcal{F}(\mathbf{x}, \{W_i\})$ represents the residual mapping to be learned.
- $W_s$ is a $1 \times 1$ projection matrix used only when matching channel dimensions or spatial strides.

### Key Advantages
1. **Unobstructed Gradient Flow**: Skip connections act as gradient highways during backpropagation because the addition operator distributes gradients directly to lower layers without scaling.
2. **Identity Fallback**: If optimal features are already learned, the network can easily push the weights of $\mathcal{F}(\mathbf{x})$ toward zero, preserving input features without performance loss.

### Block Designs

```text
BasicBlock (ResNet-18 / 34)            Bottleneck Block (ResNet-50 / 101 / 152)
---------------------------            ----------------------------------------
      Input [C, H, W]                         Input [C, H, W]
             │                                       │
       3x3 Conv, BN, ReLU                      1x1 Conv (Reduce channels)
             │                                       │
       3x3 Conv, BN                            3x3 Conv (Spatial features)
             │                                       │
             │  ┌── Skip                             1x1 Conv (Restore channels)
             ▼  │                                    │
        [+] ◄───┘                                    │  ┌── Skip
             │                                       ▼  │
           ReLU                                 [+] ◄───┘
                                                     │
                                                   ReLU
```

- **BasicBlock**: Consists of two $3 \times 3$ convolutional layers with Batch Normalization and ReLU activations. Used in shallower networks (ResNet-18, ResNet-34).
- **Bottleneck Block**: Uses a $1 \times 1 \rightarrow 3 \times 3 \rightarrow 1 \times 1$ structure to compress channel dimensions, apply spatial convolutions, and restore the channel depth. This dramatically reduces computational cost for deep architectures (ResNet-50, ResNet-101, ResNet-152).

---

## Project Structure

```text
resnet_from_scratch/
│
├── requirements.txt    # Library dependencies
├── model.py            # Modular ResNet architecture (BasicBlock, Bottleneck, ResNet)
├── dataset.py          # Data augmentation pipeline and loader setup (CIFAR-10)
├── train.py            # Training loop, evaluation, and scheduler
└── README.md           # Documentation and run instructions
```

---

## Requirements & Installation

### Dependencies
- Python $\ge 3.8$
- PyTorch $\ge 2.0.0$
- torchvision $\ge 0.15.0$
- tqdm $\ge 4.65.0$
- matplotlib $\ge 3.7.0$

### Setup Instructions
1. Create and enter the project folder:
   ```bash
   mkdir resnet_from_scratch && cd resnet_from_scratch
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

### 1. Train ResNet-18 (Default)
```bash
python train.py --arch resnet18 --epochs 20 --batch-size 128 --lr 0.1
```

### 2. Train ResNet-34
```bash
python train.py --arch resnet34 --epochs 20 --batch-size 128 --lr 0.1
```

### 3. Train ResNet-50
```bash
python train.py --arch resnet50 --epochs 30 --batch-size 64 --lr 0.05
```

---

## Expected Results & Metrics

When trained on CIFAR-10 with SGD (momentum = 0.9, weight decay = 5e-4) and Cosine Annealing learning rate scheduling:

| Model Architecture | Parameters | Train Acc (20 Epochs) | Test Acc (20 Epochs) |
| :--- | :--- | :--- | :--- |
| **ResNet-18** | ~11.2M | ~97.3% | **~97.27%** | **91.97%** |

---

## References
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770-778.