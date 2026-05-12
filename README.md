# ECGRecover: 12-Lead ECG Completion & Extension

Deep learning model (1D U-Net + Cross-Lead Attention) for restoring missing ECG leads and extending partial recordings to standard 10 s / 500 Hz format.

This project was developed as the core component for [RhythmLens](https://github.com/Akadza/RhythmLens).

## Features

- Reconstruction of any number of missing leads using physiological inter-lead dependencies (Einthoven’s Law)
- Extension of short or fragmented recordings to full 10-second, 500 Hz format
- Cross-lead attention mechanism that captures real cardiac relationships
- Support for real-world paper ECG layouts (4×3, 6×2, 6×1, rhythm strips, etc.)
- Ready-to-use training and inference pipelines

## Architecture

- **Backbone**: 1D U-Net with dilated convolutions in the bottleneck
- **Cross-Lead Attention**: Inter-lead attention module to model physiological dependencies between leads
- **Loss Function**: Weighted combination of MSE, Pearson correlation, frequency-domain loss, and Einthoven’s law physical constraints

## Training

- **Dataset**: PTB-XL (21,837 12-lead ECG records, 10 s, 500 Hz)
- **Data Splitting**: Official `strat_fold` (folds 1–8 for training, fold 9 for validation)
- **Augmentations**: Random amplitude scaling and additive noise
- **Masking Strategy**: 8 layout types simulating real paper ECG scans

### Results

| Metric               | Value      |
|----------------------|------------|
| Val PCC (Best)       | 0.9136     |
| Model Parameters     | 17.2M      |
| Training Time        | ~2 hours (NVIDIA Tesla T4) |

## Setup

```bash
git clone https://github.com/Akadza/ECGRecover.git
cd ECGRecover
pip install -r requirements.txt
```

1. Update paths.ptbxl_path in config.yaml
2. Start training:

```bash
python train.py --config config.yaml
```

Weights will be saved to **./weights/ecgrecover_best.pt.**

## Inference
Place any number of partial .csv files in ./data/input/, then run:
```bash
python infer.py --config config.yaml
```

Results (reconstructed CSV + 12-lead PNG plots) will appear in ./data/output/.
The output folder is automatically cleaned before each run.

## Project Structure
├── config.yaml              # All paths, hyperparameters, masking settings
├── train.py                 # Full training pipeline
├── infer.py                 # Batch inference and visualization
├── model/
│   └── unet.py              # ECGRecoverUNetV2 architecture
├── weights/                 # Stored checkpoints
├── data/
│   ├── input/               # Place your partial ECG files here
│   └── output/              # Results (CSV + PNG)
└── requirements.txt