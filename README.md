# CNN vs Vision Transformer on EuroSAT 🛰️

*Mini Project for Deep Learning and Neural Networks with Applications · ImageNet-pretrained ResNet-50 and ViT-B/32 fine-tuned for satellite land-cover classification*

## Key Takeaways ⭐

- Fine-tuned ResNet-50 achieved 98.14% test accuracy, the strongest baseline on the EuroSAT dataset.
- ViT-B/32 proved dramatically more robust to Gaussian noise, retaining 82.6% accuracy at severity 0.30 versus CNN's 11.7%.
- The project compares two families of vision architectures — CNN and Transformer — under a unified experimental pipeline.
- Transfer learning with full fine-tuning outperformed frozen-backbone training by up to 16.5 percentage points.
- Experiments span model accuracy, data efficiency, robustness to corruption, hyperparameter sensitivity, and augmentation ablations.

## Motivation ⚙️

### Chosen data-related problem

> **"How do CNN and Vision Transformer architectures compare for satellite land-cover classification in terms of accuracy, robustness, data efficiency, and computational cost?"**

Land-cover classification from satellite imagery is a core remote sensing task with applications in agriculture, urban planning, and environmental monitoring. This project benchmarks a classic CNN (ResNet-50) against a Vision Transformer (ViT-B/32) on the EuroSAT dataset, evaluating not just raw accuracy but also robustness to image corruption, sensitivity to hyperparameters, and the effect of transfer learning and data augmentation.

### Chosen dataset

The dataset used was [EuroSAT](https://github.com/phelber/EuroSAT), consisting of 27,000 RGB satellite images at 64×64 pixel resolution covering 10 land-cover classes: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, and SeaLake. The dataset was split 70/15/15 (18,900 train / 4,050 validation / 4,050 test). All images were resized to 224×224 and normalised with ImageNet mean and standard deviation.

### Chosen deep learning algorithms

| Model | Why I chose it |
|---|---|
| **ResNet-50 (CNN)** | Deep residual network with skip connections; a strong, widely-used CNN baseline for image classification |
| **ViT-B/32 (Vision Transformer)** | Transformer-based architecture that splits images into patches; represents the modern alternative to CNNs for vision tasks |

Both models were loaded with ImageNet-pretrained weights from `torchvision.models` and fine-tuned on the EuroSAT training set. Training used AdamW optimiser with a learning rate of 5e-5, cross-entropy loss, early stopping (patience=3), and a batch size of 64 across 8 epochs. Each experiment was run over two random seeds (42, 123) and results averaged.

## Project Objectives 🎯

1. Establish baseline classification performance for ResNet-50 and ViT-B/32 on EuroSAT.
2. Measure data efficiency by training on 50% and 10% subsets of the training data.
3. Evaluate robustness to image corruption (Gaussian noise and Gaussian blur) at multiple severity levels.
4. Conduct hyperparameter sweeps over learning rate, weight decay, and batch size.
5. Run ablation studies comparing frozen vs. fine-tuned backbones and training with vs. without data augmentation.

## Environment 👩🏻‍💻

<p align="center">
  <img src="https://img.shields.io/badge/jupyter-F37626?style=flat&logo=jupyter&logoColor=white" alt="Jupyter"/>
  <img src="https://img.shields.io/badge/Google%20Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white" alt="Google Colab"/>
  <img src="https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white" alt="GitHub"/>
</p>

## Stack 🛠️

<p align="center">
  <img src="https://img.shields.io/badge/python-3776AB?style=flat&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/torchvision-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="torchvision"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white" alt="pandas"/>
  <img src="https://img.shields.io/badge/numpy-013243?style=flat&logo=numpy&logoColor=white" alt="numpy"/>
  <img src="https://img.shields.io/badge/matplotlib-11557C?style=flat" alt="matplotlib"/>
  <img src="https://img.shields.io/badge/seaborn-4C72B0?style=flat" alt="seaborn"/>
  <img src="https://img.shields.io/badge/Pillow-9D6E33?style=flat" alt="Pillow"/>
</p>

## Repository Structure 🌲

```
.
├──.gitattributes
├── Project.ipynb
└── README.md
```

## Method 🧪

| Stage | What I did |
|---|---|
| **Setup** | Imported PyTorch, torchvision, scikit-learn; set random seeds (42, 123); configured device (CPU/CUDA/MPS) |
| **Data loading** | Downloaded EuroSAT via `torchvision.datasets`; resized to 224×224; normalised with ImageNet statistics |
| **Data split** | 70/15/15 train-validation-test split (18,900 / 4,050 / 4,050) |
| **Model building** | Loaded ResNet-50 and ViT-B/32 with ImageNet weights; replaced classification heads for 10 classes |
| **Baseline training** | Fine-tuned both models over 8 epochs with AdamW (lr=5e-5), early stopping (patience=3), batch size=64 |
| **Data efficiency** | Trained on 50% and 10% subsets to measure performance under reduced data |
| **Robustness** | Evaluated on test images corrupted with Gaussian noise (0.05–0.30) and Gaussian blur (radius 1–6) |
| **Hyperparameter sweeps** | Swept learning rate (1e-4, 5e-5, 1e-5), weight decay (0.0, 0.01, 0.1), and batch size (32, 64) |
| **Ablations** | Compared frozen vs. fine-tuned backbone; training with vs. without random flip/rotation augmentation |
| **Evaluation** | Measured accuracy, macro-F1, confusion matrices, parameter count, training time, and inference latency |

## Results 📊

### Baseline Performance (mean ± std over 2 seeds)

| Model | Test Accuracy | Macro-F1 | Parameters | Training Time | Latency (ms/img) |
|---|---|---|---|---|---|
| **ResNet-50 (CNN)** | 0.9814 ± 0.003 | 0.9809 ± 0.003 | 23.5M | 1404 s | 3.35 |
| **ViT-B/32 (Transformer)** | 0.9732 ± 0.005 | 0.9725 ± 0.005 | 87.5M | 1375 s | 3.26 |

- **ResNet-50 achieved the highest accuracy** while using 3.7× fewer parameters than ViT-B/32.
- Both models converged within 7–8 epochs, with ResNet-50 reaching ~99% training accuracy and ViT-B/32 ~99.4%.

### Data Efficiency

| Model | 100% Data | 50% Data | 10% Data |
|---|---|---|---|
| **ResNet-50** | 0.9814 | 0.9725 | 0.9485 |
| **ViT-B/32** | 0.9732 | 0.9670 | 0.9572 |

- ResNet-50 dropped 3.3 percentage points from full to 10% data, while ViT-B/32 dropped only 1.6 points, suggesting better data efficiency for the Transformer at low-data regimes.

### Robustness to Corruption

| Model | Noise 0.05 | Noise 0.10 | Noise 0.20 | Blur r=1 | Blur r=2 | Blur r=4 |
|---|---|---|---|---|---|---|
| **ResNet-50** | 0.631 | 0.278 | 0.105 | 0.201 | 0.213 | 0.170 |
| **ViT-B/32** | 0.972 | 0.959 | 0.894 | 0.617 | 0.540 | 0.330 |

- ViT-B/32 was dramatically more robust to Gaussian noise, retaining 82.6% accuracy even at severity 0.30 where ResNet-50 collapsed to 11.7%.
- ViT-B/32 also retained more accuracy under blur, though both models degraded sharply at higher radii.

### Ablation Highlights

| Ablation | ResNet-50 | ViT-B/32 |
|---|---|---|
| **Frozen backbone** | 0.817 | 0.902 |
| **Full fine-tuning** | 0.979 | 0.974 |
| **Augmentation off** | 0.982 | 0.970 |
| **Augmentation on** | 0.980 | 0.974 |

- Full fine-tuning improved accuracy by 16.5 and 7.2 percentage points for ResNet-50 and ViT-B/32 respectively over frozen backbones.
- Data augmentation slightly hurt ResNet-50 but provided a small benefit to ViT-B/32.

## Main Findings 🔍

- **ResNet-50 was the stronger baseline** on clean EuroSAT images, edging out ViT-B/32 by ~0.8 percentage points while using far fewer parameters.
- **ViT-B/32 was substantially more robust to corruption**, particularly Gaussian noise, where it maintained high accuracy even as ResNet-50 collapsed.
- **Transfer learning matters**: full fine-tuning of the pretrained backbone was critical for both architectures, with frozen-backbone training causing large accuracy drops.
- **ViT-B/32 was more data-efficient** at the 10% data regime, losing only 1.6 percentage points versus ResNet-50's 3.3.
- **Hyperparameter sensitivity differed**: ResNet-50 preferred a higher learning rate (1e-4), while ViT-B/32 performed best at 1e-5.
- **Augmentation effects were architecture-dependent**: random flips and rotations slightly reduced ResNet-50 accuracy but improved ViT-B/32.

## Recommendations for Improvements 📈

- **Cross-validation** with more seeds for more reliable estimates of model comparison.
- **Hyperparameter tuning** via grid or random search, including learning-rate scheduling and warmup for ViT.
- **Stronger augmentation** strategies such as MixUp, CutMix, or RandAugment, especially for ViT-B/32.
- **Robustness training** by including corrupted images during training to improve CNN resilience to noise and blur.
- **Additional architectures** such as ResNet-101, ViT-L/16, or EfficientNet for a broader comparison.

## Reflection 🪞

This mini project gave me hands-on experience with the full deep learning workflow — from data loading and preprocessing to fine-tuning pretrained models, designing ablation studies, and evaluating architectures beyond raw accuracy. The most striking finding was how differently CNNs and Vision Transformers behave under input corruption: ResNet-50 was the stronger classifier on clean images, yet ViT-B/32 was far more robust to noise. This reinforced that model selection depends on the deployment context, not just leaderboard accuracy. Next, I want to explore learning-rate scheduling, stronger augmentation, and additional architectures to deepen the comparison.
