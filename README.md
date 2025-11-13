# Hybrid Nanofluid Boundary-Layer ML Predictor

Predict the velocity (`f3 = f''(η)`) and temperature (`f5 = θ'(η)`) gradients of a hybrid nanofluid boundary layer by mimicking the RK4 + shooting solver with physics-informed data handling.

## Repository structure

| Path | Purpose |
| --- | --- |
| `AGENT.md` | Agent specification / FPF contract. |
| `survey_sample_data.xlsx` | Raw solver snapshots (inputs, gradients). |
| `clean_dataset.csv` | Cleaned master dataset (8 inputs → 2 outputs). |
| `data/processed/train_dataset.csv` | Training split (created by `preprocess.py` or `01_data_cleaning.ipynb`). |
| `data/processed/test_dataset.csv` | Held-out test split. |
| `models/classical_baseline.pkl` | Best-performing classical pipeline (currently XGBoost-based). |
| `models/neural_network.pt` | PyTorch network weights (64→128→64→2). |
| `models/neural_network_metadata.json` | Normalization stats + metrics for the NN. |
| `notebooks/01_data_cleaning.ipynb` | Loads Excel, applies FPF cleaning rules, saves CSV/splits. |
| `notebooks/02_classical_models.ipynb` | RF/GBR/KNN/XGBoost/LightGBM baselines + plots. |
| `notebooks/03_neural_network.ipynb` | PyTorch training with early stopping + diagnostic curves. |
| `preprocess.py` | Headless cleaner + splitter. |
| `train_model.py` | CLI training entrypoint (classical or neural). |
| `model_inference.py` | Unified inference CLI for both model families. |
| `reports/verification_summary.json` | JSON summary of verification metrics/checks. |
| `reports/*.png` | Parity and residual plots for diagnostics. |

## Data workflow (FPF-compliant)

1. **Clean** raw Excel (`survey_sample_data.xlsx`) by either executing `notebooks/01_data_cleaning.ipynb` or running:
   ```bash
   python preprocess.py
   ```
   This enforces mandatory rules from `AGENT.md`: drop NaNs, remove |f3|/|f5| > 1e4 blow-ups, convert `S`>3.5 (degrees) to radians, keep `0 ≤ η ≤ 5`, and restore canonical column ordering `[M, S, K, phi1, phi2, Ec, Pr, eta, f3, f5]`.
2. **Outputs**: `clean_dataset.csv`, `data/processed/train_dataset.csv`, `data/processed/test_dataset.csv`.

## Modeling

- **Classical baselines** – See `notebooks/02_classical_models.ipynb`. The notebook evaluates RandomForest, GradientBoosting, KNN, **XGBoost**, and **LightGBM** under deterministic settings and saves the best pipeline to `models/classical_baseline.pkl`. Reproduce via CLI if needed:
  ```bash
  python train_model.py --model-type classical --estimator random_forest
  python train_model.py --model-type classical --estimator xgboost
  python train_model.py --model-type classical --estimator lightgbm
  ```
- **Neural network** – `notebooks/03_neural_network.ipynb` trains the mandated 64→128→64→2 dense net with Adam, early stopping (patience 40), and learning-curve plots. Artifacts land in `models/neural_network.pt` plus `models/neural_network_metadata.json`. Optional CLI bake:
  ```bash
  python train_model.py --model-type neural
  ```

Both notebooks were executed via `nbclient` in this workspace so the saved artifacts already reflect their outputs.

## Inference

Use `model_inference.py` for single samples or CSV batches. The script auto-converts `S` values above 3.5 to radians to stay faithful to the solver physics.

### Example (inline values)
```bash
python model_inference.py --model-type classical --values 0.5 60 1 0.02 0.01 0.5 204 0.333
python model_inference.py --model-type neural --values 0.5 60 1 0.02 0.01 0.5 204 0.333
```

### Example (CSV batch)
```bash
python model_inference.py --model-type classical --input-csv samples.csv
```
`samples.csv` must contain all eight feature columns in the canonical order.

## Verification & validation

All verification steps are summarized here; raw outputs live in `reports/verification_summary.json` and PNG plots in `reports/`.

- **Data leakage** – Comparing full `(M, S, K, phi1, phi2, Ec, Pr, eta, f3, f5)` tuples, train and test sets have `leakage_count = 0`.
- **Units sanity** – In the cleaned dataset, `S` lies in `[0.524, 1.047]` radians (≈30°–60°), confirming all angles are in radians after preprocessing.
- **Outliers** – Maximum absolute target magnitude is `|f3,f5| ≤ 5.65` and maximum absolute prediction on the test set is ≈2.51, far below the 1e4 threshold used to detect numerical blow-ups.

### Test metrics (best classical model: XGBoost)

The current `models/classical_baseline.pkl` is an XGBoost-based pipeline; on the held-out test set it achieves approximately:

- `f3` (velocity gradient): MAE ≈ 0.095, RMSE ≈ 0.159, R² ≈ 0.73  
- `f5` (temperature gradient): MAE ≈ 0.33, RMSE ≈ 0.74, R² ≈ 0.89  
- Joint: MAE ≈ 0.21, RMSE ≈ 0.54

The PyTorch neural network reproduces similar trends for `f3` but underperforms for `f5` (higher RMSE, lower R²); its exact metrics are stored in `models/neural_network_metadata.json` and `reports/verification_summary.json`.

### Residual diagnostics

Parity and residual plots for the classical model are saved as:

- `reports/classical_pred_vs_actual_f3.png`, `reports/classical_pred_vs_actual_f5.png`  
- `reports/classical_residuals_f3.png`, `reports/classical_residuals_f5.png`

Observations:

- `f3` residuals are roughly zero-mean with no strong correlation against predictions (no obvious systematic bias).  
- `f5` residuals show a mild positive correlation with predictions (corr(pred,resid) ≈ 0.40), indicating remaining systematic error in certain regions of the temperature-gradient field.

### Train vs test vs cross-validation

From `reports/verification_summary.json`:

- Train/test metrics for the classical LightGBM-based model and the neural net are stored under `classical_*` and `neural_*`.  
- A 5-fold cross-validation using XGBoost on the full cleaned dataset yields:
  - MAE ≈ 0.14 ± 0.06, RMSE ≈ 0.43 ± 0.20  
  - R²(f3) ≈ 0.82 ± 0.09, R²(f5) ≈ 0.88 ± 0.12
- Held-out test performance for the XGBoost baseline is broadly consistent with these CV statistics, suggesting no severe overfitting, though `f5` remains more sensitive to data distribution shifts than `f3`.

### Train–test distribution comparison

Also from `reports/verification_summary.json`:

- Most feature means (M, S, K, phi1, Ec) are closely matched between train and test.  
- Two mild shifts are present:
  - `phi2`: train mean ≈ 0.0121, test mean ≈ 0.0106  
  - `eta`: train mean ≈ 1.36, test mean ≈ 1.66

These shifts likely contribute to the harder generalization of `f5` in some regions, and should guide future data collection.

### Overall verdict

- Preprocessing, unit handling, and splitting follow the AGENT specification and FPF rules (no leakage, no solver blow-ups, consistent units).  
- The XGBoost classical model is the recommended predictor for `(f3, f5)` on the current dataset.  
- The temperature gradient `f5` still exhibits structured residuals; to improve, expand the dataset in under-represented `eta`/`phi2` regimes and consider re-weighting or retuning models with a stronger focus on `f5`.

## Determinism & notes

- Seeds (`42`) are fixed for NumPy, PyTorch, and scikit-learn where applicable.
- The PyTorch checkpoint bundles feature means/stds; the metadata JSON mirrors those values for transparency.
- Training outside notebooks is possible via `train_model.py`, but day-to-day experimentation should continue inside the `.ipynb` files to respect the user directive.
