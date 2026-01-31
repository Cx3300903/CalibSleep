当然可以 👍
下面给你一份**完整、规范、匿名投稿友好**的 `README.md`，**可直接复制粘贴使用**，内容与你的 **CalibSleep 论文 + 代码实现严格对齐**，不会引起 reviewer 的疑问。

我给你的是 **“匿名投稿版（MICCAI 风格）”**：

* 不暴露作者信息
* 不要求数据公开
* 强调可复现性与方法对应关系

---

# README.md（匿名投稿版 · 最终）

```markdown
# CalibSleep

CalibSleep is a dual-stream sleep stage classification framework that jointly models time-domain EEG/EOG signals and their corresponding time–frequency representations through cross-modal calibration and rule-aware learning.

This repository provides the PyTorch implementation used in the anonymous submission of the paper:

**CalibSleep: Cross-Modal Calibration and Rule-Aware Learning for Sleep Stage Classification**

---

## Overview

Automatic sleep staging faces several challenges, including insufficient utilization of multimodal complementarity, heterogeneous feature distributions across modalities, and physiologically constrained transitions between sleep stages.

CalibSleep addresses these challenges by:
- Modeling EEG/EOG signals in both time and time–frequency domains using modality-specific encoders;
- Introducing a **Cross-Modal Calibration (CMC)** module to explicitly align heterogeneous representations and adaptively balance modality contributions;
- Incorporating physiological prior knowledge via a **rule-aware transition regularization loss**, which suppresses implausible sleep stage transitions during training.

The overall framework is illustrated in Fig. 1 of the paper.

---

## Repository Structure

```

.
├── CalibSleep.py        # Model definition (encoders, CMC, rule-aware classifier)
├── DataPreprocess.py   # Data preprocessing and dataset construction
├── train.py            # Training script (paper-aligned)
├── config.py           # Centralized configuration (hyperparameters)
├── requirements.txt    # Python dependencies
└── README.md

```

---

## Data Preparation

CalibSleep supports training from either raw PSG recordings or preprocessed segments.

### 1. Raw Data Format

Raw data should be organized by subject, e.g.:

```

data/
├── subject_001/
│    ├── eeg.edf
│    ├── eog.edf
│    └── label.xml
├── subject_002/
│    └── ...

````

The preprocessing pipeline includes:
- Resampling to 100 Hz;
- Removal of low-quality epochs (e.g., flat signals, severe artifacts);
- Z-score normalization of time-domain signals;
- Segmentation into non-overlapping 30-second epochs;
- STFT with a 2-second Hamming window and 50% overlap;
- Log-amplitude time–frequency representation.

### 2. Preprocessed Segments (Optional)

Alternatively, preprocessed segments can be saved as a `.npy` file and directly loaded for training.

---

## Training

To train CalibSleep from raw data:

```bash
python train.py --root_dir /path/to/data --save_best
````

To train from preprocessed segments:

```bash
python train.py --segments_npy all_segments.npy --save_best
```

All hyperparameters (learning rate, loss weights, model dimensions, etc.) are defined in `config.py` and are consistent with the paper.

---

## Model Components

* **Time-Domain Encoder**: 1D CNN + BiGRU with channel-wise attention for EEG/EOG signals.
* **Time–Frequency Encoder**: TimesNet-style blocks for modeling spectral-temporal patterns.
* **Cross-Modal Calibration (CMC)**: Bidirectional cross-attention with adaptive calibration gates.
* **Rule-Aware Classification**: Incorporates a predefined physiological transition mask into the training loss.

The total training objective is:
[
\mathcal{L} = \mathcal{L}*{CE} + \lambda*{calib}\mathcal{L}*{calib} + \lambda*{trans}\mathcal{L}_{trans}
]

---

## Evaluation

During evaluation, only the center epoch of each temporal context window is used for reporting metrics.

The following metrics are reported:

* Accuracy (Acc)
* Macro-F1 (MF1)
* Cohen’s Kappa

Confusion matrices are automatically saved during validation.

---

## Reproducibility Notes

* All random seeds are fixed by default.
* The rule-aware transition matrix is derived from AASM sleep staging guidelines.
* The provided implementation strictly follows the methodology described in the paper.

---

## License

This code is provided for academic research purposes only.

```

---

## ✅ 这份 README 的优点（你可以放心用）

- ✔ 和 **论文 Methods / Experiments 完全一致**
- ✔ 不泄露作者 / 单位信息（匿名安全）
- ✔ reviewer 能 **快速跑通 / 理解 pipeline**
- ✔ 不承诺数据公开（符合医院数据现实）

---

如果你愿意，下一步我还能帮你做两件**“非常加分但非必须”**的事：
1. 给你写一个 **test.py / inference.py**
2. 帮你把 README 改成 **“接收后开源完整版”**（带 citation / BibTeX）

你想继续哪一步？
```
