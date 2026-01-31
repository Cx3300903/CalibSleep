# CalibSleep

CalibSleep is a dual-stream sleep stage classification framework that jointly models time-domain EEG/EOG signals and their corresponding time–frequency representations through cross-modal calibration and rule-aware learning.

This repository provides the PyTorch implementation used in the anonymous submission of the paper:

**CalibSleep: Cross-Modal Calibration and Rule-Aware Learning for Sleep Stage Classification**

---

## Overview

Automatic sleep stage classification is challenging due to the heterogeneous characteristics of physiological signals and the complex transition dynamics between sleep stages. Existing methods often rely on a single representation domain or simple feature fusion strategies, which limits their ability to exploit complementary information and enforce physiological consistency.

CalibSleep addresses these challenges by:
- Jointly modeling **time-domain signals** and **time–frequency representations**,
- Introducing a **Cross-Modal Calibration (CMC)** module to align heterogeneous features and dynamically balance their contributions,
- Incorporating **physiological stage transition priors** into training via a rule-aware regularization strategy.

---

## Model Architecture

CalibSleep consists of four main components:

1. **Time-Domain Encoder**  
   Multi-scale 1D CNN combined with channel attention and BiGRU to extract temporal features from raw EEG/EOG signals.

2. **Time–Frequency Encoder**  
   TimesNet-style blocks with FFT-based periodic enhancement and multi-scale temporal convolutions to model rhythmic patterns from time–frequency representations.

3. **Cross-Modal Calibration Module (CMC)**  
   Bidirectional cross-attention aligns time-domain and time–frequency features, followed by temporal pooling and adaptive calibration fusion.

4. **Rule-Aware Classification Head**  
   A lightweight classifier predicts sleep stages while a transition regularization term penalizes physiologically implausible stage transitions.

---

## Repository Structure
.
├── CalibSleep.py        # Model definition (encoders, CMC, rule-aware classifier)
├── DataPreprocess.py   # Data preprocessing and dataset construction
├── train.py            # Training script (paper-aligned)
├── requirements.txt    # Python dependencies
└── README.md

---
## Data Preparation

CalibSleep supports training from either raw PSG recordings or preprocessed segments.

### Raw Data Format

Raw data should be organized by subject as follows:
data/
├── subject_001/
│   ├── eeg.edf
│   ├── eog.edf
│   └── label.xml
├── subject_002/
│   └── ...
