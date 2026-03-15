<div align="center">

<!-- HEADER BANNER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0A4C6A,1A7A5E&height=200&section=header&text=AHR-KAN&fontSize=80&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Morphology-Aware%20Depth%20Estimation%20for%20Seaweed%20Biomass%20Prediction&descAlignY=58&descSize=16" width="100%"/>

<br/>

<!-- BADGES -->
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![ITS](https://img.shields.io/badge/Institut-Teknologi%20Sepuluh%20Nopember-E76F51?style=for-the-badge)](https://www.its.ac.id)

<br/>

> **Morphology Aware Depth Estimation for *Eucheuma cottonii* Biomass Prediction Using Adaptive Hybrid Kolmogorov-Arnold Networks**
>
> *Engineering Applications of Artificial Intelligence (EAAI), Elsevier — Under Review*

<br/>

</div>

---

## 📖 Overview

This repository provides the official implementation of **AHR-KAN** (Adaptive Hybrid Regression via Kolmogorov-Arnold Networks), a multi-modal framework for non-destructive dry biomass estimation of the red seaweed species *Eucheuma cottonii*.

The framework overcomes the fundamental limitation of 2D-only approaches by recovering **3D morphological structure** from standard RGB images — without depth sensors — through physics-guided camera geometry and monocular depth estimation. Seven novel hybrid ensemble architectures then fuse these depth-aware features with predictions from diverse base regressors using adaptive KAN-based meta-learning.

---

## ✨ Key Contributions

1. **Physics-Guided Metric Scaling** — Anchors monocular depth estimates to real-world units using the pinhole camera model, resolving scale ambiguity without depth sensors.

2. **Depth-Aware 8-Feature Morphological Descriptor** — Combines 2D geometric measurements (area, length, width, perimeter, solidity, aspect ratio) with 3D volumetric attributes (thickness, volume) derived from absolute depth maps.

3. **AHR-KAN: First KAN for Hybrid Meta-Learning** — Introduces the Kolmogorov-Arnold representation theorem to the stacking ensemble paradigm. AHR-KAN learns context-dependent predictor fusion via adaptive univariate basis functions, weighting base model outputs according to morphological characteristics.

4. **AHR-KAN-HighOrder** — Extends AHR-KAN with 27 orthogonal basis functions per input dimension (Chebyshev deg-10 + Fourier freq-6 + Gaussian RBF ×10) for highly irregular specimens.

5. **AHR-KAN-Uncertainty** — Probabilistic extension with dual prediction heads (mean + variance) trained via heteroscedastic loss, enabling calibrated 95% confidence intervals for risk-aware harvest decisions.

6. **Four Additional Hybrid Architectures**: CCAN (cross-covariance attention), DGFC (dynamic graph feature convolution), RECN (residual error-correcting network), and DERH (deep evidential regression hybrid).

---

## ⚙️ Installation

### Prerequisites

- Python 3.9+
- CUDA-enabled GPU (recommended for DL models)
- 8 GB+ RAM

### Clone and Install

```bash
git clone https://github.com/YOUR_USERNAME/ahr-kan-seaweed-biomass.git
cd ahr-kan-seaweed-biomass
```

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```txt
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
shap>=0.42.0
scipy>=1.11.0
opencv-python>=4.8.0
ultralytics>=8.0.0
Pillow>=10.0.0
tqdm>=4.65.0
```
---

## 🤖 Models

### Baseline Regressors

| Model | Type | Key Config |
|-------|------|-----------|
| **Random Forest** | Tree Ensemble | 200 trees, max_depth=10, min_samples_leaf=2 |
| **XGBoost** | Gradient Boosting | 200 est., η=0.1, max_depth=6, λ=1.0 |
| **LightGBM** | Gradient Boosting | 200 est., η=0.1, leaf-wise growth |
| **SVR** | Kernel Method | RBF kernel, C=10, ε=0.1 |
| **1D-CNN** | Deep Learning | Conv1D(32,64), dropout=0.3 |
| **Bi-LSTM** | Deep Learning | 2-layer, hidden=64, dropout=0.3 |
| **Transformer** | Deep Learning | 2-layer, 4 heads, d_model=64 |

### Novel Hybrid Architectures

| Model | Mechanism | Uncertainty | R² (mean±std) |
|-------|-----------|:-----------:|:-------------:|
| **AHR-KAN** ⭐ | Adaptive KAN basis functions | — | **0.9623 ± 0.017** |
| **AHR-KAN-HighOrder** | Chebyshev + Fourier + RBF (27 basis) | — | 0.9570 ± 0.018 |
| **AHR-KAN-Uncertainty** | Dual-head KAN + heteroscedastic loss | Aleatoric | 0.9558 ± 0.017 |
| **CCAN** | Cross-covariance attention | — | 0.9531 ± 0.016 |
| **DGFC** | Dynamic graph convolution | — | 0.9513 ± 0.018 |
| **DERH** | Normal-Inverse-Gamma prior | Both | 0.9482 ± 0.017 |
| **RECN** | Residual error correction | — | 0.9469 ± 0.017 |

---

## 🙏 Acknowledgements

This research was supported by:

- 🇮🇩 **Indonesian Endowment Fund for Education (LPDP)**, Ministry of Higher Education, Science and Technology — **EQUITY Program** (Contract No. 4299/B3/DT.03.08/2025 & No. 3029/PKS/ITS/2025)
- 🔬 **National Research and Innovation Agency (BRIN)** — RIIM Kompetisi Gelombang 10 Program
- 🏛️ **Department of Informatics, Institut Teknologi Sepuluh Nopember (ITS)**, Surabaya, Indonesia

Data collection was conducted at seaweed cultivation sites in **Takalar (South Sulawesi)** and **Mamuju (West Sulawesi)**, Indonesia.

---

## 📬 Contact

| Author | Role | Email |
|--------|------|-------|
| **Rumaisha Afrina** | First Author | rumaisha.afrina@gmail.com |

Department of Informatics, Institut Teknologi Sepuluh Nopember
Surabaya 60111, East Java, Indonesia

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0A4C6A,1A7A5E&height=100&section=footer" width="100%"/>

*Made with 🌿 for sustainable seaweed aquaculture · Institut Teknologi Sepuluh Nopember*

</div>
