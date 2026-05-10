# ECGRecover: Reconstruction of 12-lead ECG from Partial Data

## Project Description

ECGRecover is a deep learning model based on a 1D U-Net architecture with a cross-lead attention mechanism, designed to reconstruct missing ECG leads from partial recordings.  
**This project was developed as the core component for [RhythmLens](https://github.com/Akadza/RhythmLens).**

## Architecture

- **Backbone**: 1D U-Net with dilated convolutions in the bottleneck  
- **Cross-Lead Attention**: Inter-lead attention mechanism that captures physiological dependencies between leads (Einthoven’s Law)  
- **Loss Function**: Weighted combination of MSE, Pearson correlation, frequency-domain loss, and physical constraints based on Einthoven’s relations

## Training

- **Dataset**: [PTB-XL](https://physionet.org/content/ptb-xl/1.0.1/) — 21,837 12-lead ECG records, 10 seconds, 500 Hz  
- **Data Splitting**: Official `strat_fold` (folds 1-8 for training, fold 9 for validation)  
- **Augmentations**: Random amplitude scaling and additive noise  
- **Masking Strategy**: 8 layout types (4×3, 6×2, 6×1, 3×1 + rhythm strips + random) to simulate real-world paper ECG scans

### 🎯 Results

| Metric                  | Value                  |
|-------------------------|------------------------|
| **Val PCC (Best)**      | **0.9136**             |
| **Model Parameters**    | 17.2M                  |
| **Training Time**       | ~1 hours GPU T4 x2     |

### Installation

```bash
pip install -r requirements.txt