# 🌊 Hybrid Nanofluid Boundary-Layer ML Predictor

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A machine learning project that predicts velocity gradients `f3 = f''(η)` and temperature gradients `f5 = θ'(η)` of hybrid nanofluid boundary layers using physics-informed models trained on MATLAB RK4 + shooting method solver data.

## 🎯 Project Overview

This project bridges computational fluid dynamics and machine learning by creating surrogate models that replicate the behavior of numerical solvers for hybrid nanofluid flow over stretching sheets. The models predict critical gradients that characterize heat transfer and momentum transport in nanofluid systems.

## 📁 Repository Structure

```
NANO-FLUIDS/
├── 📄 AGENT.md                          # Agent specification & FPF contract
├── 📄 README.md                         # This file
├── 📄 report.md                         # Detailed technical report
│
├── 📊 Data Files
│   ├── survey_sample_data.xlsx          # Raw MATLAB solver snapshots
│   ├── clean_dataset.csv                # Cleaned master dataset (8 inputs → 2 outputs)
│   └── data/processed/
│       ├── train_dataset.csv            # Training split (80%)
│       └── test_dataset.csv             # Held-out test split (20%)
│
├── 🤖 Models
│   ├── models/classical_baseline.pkl    # Best XGBoost pipeline
│   ├── models/neural_network.pt         # PyTorch network (64→128→64→2)
│   └── models/neural_network_metadata.json  # Normalization stats & metrics
│
├── 📓 Notebooks
│   ├── notebooks/01_data_cleaning.ipynb      # Data preprocessing & FPF rules
│   ├── notebooks/02_classical_models.ipynb   # Classical ML baselines
│   └── notebooks/03_neural_network.ipynb     # Neural network training
│
├── 🔧 Scripts
│   ├── preprocess.py                    # Automated data cleaner & splitter
│   ├── train_model.py                   # CLI training entrypoint
│   ├── model_inference.py               # Unified inference interface
│   └── visualize.py                     # Visualization utilities
│
└── 📈 Reports
    ├── reports/verification_summary.json     # Comprehensive metrics
    ├── reports/classical_parity_*.png        # Parity plots
    ├── reports/classical_residuals_*.png     # Residual diagnostics
    ├── reports/neural_*.png                  # Neural network diagnostics
    ├── reports/correlation_matrix.png        # Feature correlations
    ├── reports/feature_distributions.png     # Distribution analysis
    └── reports/cross_validation_summary.png  # CV results
```

## 🔬 Physics Background

### Governing System
This project models boundary-layer flow of hybrid nanofluids over stretching sheets. The MATLAB solver computes:

- **f1** = f (stream function)
- **f2** = f′ (velocity)
- **f3** = f′′ (velocity gradient) ← **ML Target 1**
- **f4** = θ (temperature)
- **f5** = θ′ (temperature gradient) ← **ML Target 2**

### Input Features (8)
| Feature | Description | Units |
|---------|-------------|-------|
| `M` | Magnetic field parameter | dimensionless |
| `S` | Rotation/angle parameter | radians |
| `K` | Permeability parameter | dimensionless |
| `phi1` | Nanoparticle volume fraction #1 | dimensionless |
| `phi2` | Nanoparticle volume fraction #2 | dimensionless |
| `Ec` | Eckert number | dimensionless |
| `Pr` | Prandtl number | dimensionless |
| `eta` | Similarity variable | dimensionless (0-5) |

### Output Targets (2)
- **f3**: Velocity gradient f′′(η) — characterizes momentum transport
- **f5**: Temperature gradient θ′(η) — characterizes heat transfer

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.8+
pip install numpy pandas scikit-learn xgboost lightgbm torch matplotlib seaborn openpyxl
```

### Installation
```bash
git clone https://github.com/BhaveshBytess/NANOFLUIDS.git
cd NANOFLUIDS
```

## 📊 Data Workflow (FPF-Compliant)

### Step 1: Data Cleaning
Run either the notebook or the CLI script:

**Option A: Jupyter Notebook**
```bash
jupyter notebook notebooks/01_data_cleaning.ipynb
```

**Option B: Command Line**
```bash
python preprocess.py
```

### Cleaning Rules (Mandatory - see AGENT.md)
1. ✅ Drop rows with missing `f3` or `f5`
2. ✅ Remove numerical blow-ups: `|f3|` or `|f5| > 10⁴`
3. ✅ Convert rotation parameter: `S > 3.5` from degrees → radians
4. ✅ Enforce domain bounds: `0 ≤ η ≤ 5`
5. ✅ Restore canonical column ordering: `[M, S, K, phi1, phi2, Ec, Pr, eta, f3, f5]`
6. ✅ Generate stratified train/test split (80/20)

### Outputs
- `clean_dataset.csv` — Master cleaned dataset
- `data/processed/train_dataset.csv` — Training split
- `data/processed/test_dataset.csv` — Test split

## 🤖 Model Training

### Classical Machine Learning Models

Train various ensemble models with automatic hyperparameter optimization:

```bash
# Random Forest
python train_model.py --model-type classical --estimator random_forest

# XGBoost (Best Performance)
python train_model.py --model-type classical --estimator xgboost

# LightGBM
python train_model.py --model-type classical --estimator lightgbm

# Gradient Boosting
python train_model.py --model-type classical --estimator gradient_boosting

# K-Nearest Neighbors
python train_model.py --model-type classical --estimator knn
```

**Recommended:** XGBoost provides the best balance of accuracy and generalization.

### Neural Network Model

Train the physics-informed deep neural network:

```bash
python train_model.py --model-type neural
```

**Architecture:**
```
Input (8 features)
    ↓
Dense(64) + ReLU
    ↓
Dense(128) + ReLU
    ↓
Dense(64) + ReLU
    ↓
Dense(2) — [f3, f5]
```

**Training Configuration:**
- Optimizer: Adam
- Loss: MSE (multi-output)
- Early stopping: patience 40 epochs
- Learning rate: adaptive
- Batch size: 32

## 🔮 Model Inference

### Single Sample Prediction

Predict gradients for specific parameter combinations:

```bash
# Classical model (XGBoost)
python model_inference.py --model-type classical \
    --values 0.5 60 1 0.02 0.01 0.5 204 0.333

# Neural network model
python model_inference.py --model-type neural \
    --values 0.5 60 1 0.02 0.01 0.5 204 0.333
```

**Input order:** `M S K phi1 phi2 Ec Pr eta`

**Note:** The script automatically converts `S > 3.5` from degrees to radians.

### Batch Prediction

Process multiple samples from CSV:

```bash
python model_inference.py --model-type classical --input-csv samples.csv
```

**CSV Format Requirements:**
- Must contain columns: `M, S, K, phi1, phi2, Ec, Pr, eta`
- Column order must match the canonical ordering
- S values can be in degrees (auto-converted)

**Output:** CSV with added columns `f3_pred` and `f5_pred`

## 📊 Model Performance

### Best Classical Model: XGBoost

**Test Set Metrics:**

| Target | MAE | RMSE | R² |
|--------|-----|------|-----|
| **f3** (velocity) | 0.095 | 0.159 | 0.731 |
| **f5** (temperature) | 0.326 | 0.740 | 0.893 |
| **Joint** | 0.210 | 0.535 | — |

**5-Fold Cross-Validation:**
- MAE: 0.14 ± 0.06
- RMSE: 0.43 ± 0.20
- R²(f3): 0.82 ± 0.09
- R²(f5): 0.88 ± 0.12

### Neural Network Performance

| Target | MAE | RMSE | R² |
|--------|-----|------|-----|
| **f3** (velocity) | 0.162 | 0.208 | 0.537 |
| **f5** (temperature) | 1.180 | 2.015 | 0.207 |
| **Joint** | 0.671 | 1.432 | — |

**Note:** Neural network requires more data or regularization to match ensemble performance.

## ✅ Verification & Validation

### Data Integrity Checks
- ✅ **Zero data leakage** — No identical samples between train/test splits
- ✅ **Unit consistency** — All `S` values in radians [0.524, 1.047] (≈30°-60°)
- ✅ **Outlier removal** — Max |f3, f5| ≤ 5.65 (well below 10⁴ threshold)
- ✅ **Domain compliance** — All `η` values within [0, 5]

### Diagnostic Plots

**Parity Plots** — Predicted vs Actual:
- `reports/classical_parity_f3.png`, `reports/classical_parity_f5.png`
- `reports/neural_parity_f3.png`, `reports/neural_parity_f5.png`

**Residual Analysis:**
- `reports/classical_residuals_f3.png` — Well-behaved, zero-mean
- `reports/classical_residuals_f5.png` — Mild systematic bias at high predictions
- `reports/neural_residuals_*.png` — Larger variance, needs improvement

**Feature Analysis:**
- `reports/correlation_matrix.png` — Feature interdependencies
- `reports/feature_distributions.png` — Train/test distribution comparison

### Known Limitations

**Distribution Shifts:**
- `phi2`: train mean ≈ 0.0121, test mean ≈ 0.0106
- `eta`: train mean ≈ 1.36, test mean ≈ 1.66

**Impact:** Slightly weaker `f5` generalization in high-`eta` / low-`phi2` regions.

**Recommendation:** Collect more solver data in under-represented parameter regimes.

## 📚 Notebooks

### Interactive Analysis

1. **01_data_cleaning.ipynb** — Data preprocessing pipeline
   - Load raw Excel data
   - Apply FPF cleaning rules
   - Generate train/test splits
   - Visualize distributions

2. **02_classical_models.ipynb** — Classical ML experiments
   - Compare 5 algorithms (RF, GBR, KNN, XGBoost, LightGBM)
   - Hyperparameter optimization
   - Cross-validation analysis
   - Export best model

3. **03_neural_network.ipynb** — Deep learning experiments
   - Architecture design
   - Training with early stopping
   - Learning curves
   - Error analysis

## 🔬 First Principles Framework (FPF)

This project strictly adheres to physics-informed ML principles:

### Core Tenets
1. **No Hallucinations** — All claims trace back to physics/data
2. **Fundamental Reasoning** — ML decisions justified by physical meaning
3. **Equation Preservation** — ODEs and boundary conditions remain inviolate
4. **Solver Fidelity** — Models mimic RK4+shooting behavior, not replace it

## 🛠️ Technical Details

### Reproducibility
- All random seeds fixed to `42` (NumPy, PyTorch, scikit-learn)
- Deterministic training modes enabled
- Exact package versions tracked

### Model Artifacts
- `classical_baseline.pkl` — Serialized XGBoost pipeline with scaler
- `neural_network.pt` — PyTorch state dict
- `neural_network_metadata.json` — Normalization parameters and metrics

### Visualization Pipeline
Run comprehensive analysis:
```bash
python visualize.py
```
Generates all plots in `reports/` directory.

## 🎯 Future Improvements

### Data Enhancement
- [ ] Expand dataset in high-`eta` regions
- [ ] Add more `phi2` variation samples
- [ ] Include extreme parameter combinations

### Model Development
- [ ] Physics-informed neural network (PINN) with PDE loss
- [ ] Ensemble meta-learner (XGBoost + Neural)
- [ ] Uncertainty quantification with Bayesian methods
- [ ] Feature engineering: explicit `η` interactions

### Deployment
- [ ] REST API for real-time predictions
- [ ] Web interface for parameter exploration
- [ ] Model versioning and A/B testing
- [ ] GPU acceleration for batch inference

## 📖 References

### Physics
- Boundary-layer flow over stretching sheets
- Hybrid nanofluid thermophysical properties
- Runge-Kutta 4th order + shooting methods

### Machine Learning
- Ensemble methods for regression
- Neural networks for multi-output prediction
- Physics-informed machine learning

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Follow FPF principles
4. Add tests for new features
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 👥 Authors

**Bhavesh** — [GitHub](https://github.com/BhaveshBytess)

## 📧 Contact

For questions or collaborations:
- GitHub Issues: [NANOFLUIDS Issues](https://github.com/BhaveshBytess/NANOFLUIDS/issues)
- Email: [10bhavesh7.11@gmail.com]

---

**Note:** This project is part of ongoing research in computational fluid dynamics and physics-informed machine learning. Results should be validated against numerical solvers before use in production applications.
