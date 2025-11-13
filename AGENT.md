# 🧠 AGENT SPECIFICATION — Hybrid Nanofluid ML Project

<div align="center">

## 📌 Project Identity

**Hybrid Nanofluid Boundary-Layer ML Predictor**

*Physics-Informed Machine Learning for Computational Fluid Dynamics*

</div>

---

## 📌 Project Goal

Develop and deploy machine learning models that accurately predict critical boundary-layer gradients in hybrid nanofluid flow systems:

* **f3(η) = f′′(η)** → Velocity gradient (momentum transport)
* **f5(η) = θ′(η)** → Temperature gradient (heat transfer)

**Input:** 8 dimensionless physical parameters  
**Output:** 2 gradient values that characterize the flow field  
**Method:** Supervised learning on MATLAB RK4 + shooting method solver data

---

# 🚀 1. FIRST PRINCIPLES FRAMEWORK (FPF)

The agent operates under strict physics-informed guidelines to ensure scientific rigor and reproducibility.

## Core FPF Rules

### 🔒 FPF Rule 1 — No Hallucinations

**Mandate:** Every claim, prediction, and decision must be traceable to:
- Governing equations (ODEs, boundary conditions)
- Experimental/solver data points
- Established physical constants
- Peer-reviewed literature

**Forbidden:**
- Inventing parameter relationships
- Fabricating numerical results
- Extrapolating beyond validated domains

---

### 🔬 FPF Rule 2 — Fundamental Reasoning

**Mandate:** All machine learning decisions must be justified by physical principles:
- Why this loss function? (MSE aligns with solver residual minimization)
- Why this architecture? (Multi-output regression preserves coupled physics)
- Why these features? (Dimensionless groups from similarity analysis)

**Requirement:** Document the physics-to-ML mapping for every choice.

---

### 🧮 FPF Rule 3 — Equation Preservation

**Mandate:** The governing system is inviolate:

```
Momentum: f''' + ff'' - (f')² + M²(S - f') + K⁻¹(S - f') = 0
Energy: θ'' + Pr·f·θ' + Pr·Ec·(f'')² = 0
```

**Forbidden Actions:**
- Modifying ODEs to "improve fit"
- Altering boundary conditions
- Changing physical constants (Pr, Ec, etc.) arbitrarily

**Permitted:** Feature engineering that respects dimensionless groups.

---

### ⚙️ FPF Rule 4 — Solver Fidelity

**Understanding:** The targets (f3, f5) originate from:
1. **Runge-Kutta 4th order** integration (numerical ODE solution)
2. **Shooting method** (iterative boundary value adjustment)
3. **Tolerance:** 1e-6 convergence criterion

**ML Role:** Create a *surrogate model* that:
- Mimics solver output distribution
- Captures parameter sensitivities
- Respects solver blow-up regimes (|f| > 10⁴ indicates divergence)

**Critical Distinction:** ML does NOT replace the physics — it accelerates prediction by learning from validated solver runs.

---

# 🌡 2. PHYSICS BACKGROUND

## Problem Domain

**System:** Steady, laminar boundary-layer flow of a hybrid nanofluid (two nanoparticle types) over a stretching sheet with:
- Magnetic field effects (Lorentz force)
- Porous medium (Darcy resistance)
- Viscous dissipation (Eckert number)
- Heat transfer (temperature-dependent properties)

## Governing Equations

The system is modeled by coupled nonlinear ODEs derived from Navier-Stokes and energy equations through similarity transformations:

### State Variables (Solver Output)
| Variable | Physical Meaning | ML Relevance |
|----------|-----------------|--------------|
| **f1 = f** | Stream function | Intermediate |
| **f2 = f′** | Dimensionless velocity | Intermediate |
| **f3 = f′′** | Velocity gradient | **🎯 ML Target** |
| **f4 = θ** | Dimensionless temperature | Intermediate |
| **f5 = θ′** | Temperature gradient | **🎯 ML Target** |

### Numerical Solution Method

**Algorithm:** Fourth-order Runge-Kutta (RK4) + Shooting Method

**Procedure:**
1. Initial guess for boundary slopes (s01, s02, s11, s12)
2. RK4 integration from η=0 to η=∞ (practically η≤5)
3. Check boundary conditions at infinity
4. Adjust guesses via Newton-Raphson (shooting)
5. Iterate until convergence (tolerance 1e-6)

**Output:** Converged profiles for f, f′, f′′, θ, θ′ at discrete η points

---

# 📊 3. DATASET STRUCTURE

## Data Source
**Origin:** MATLAB solver sweeps over parameter space  
**Format:** Excel file (`survey_sample_data.xlsx`)  
**Size:** ~3000 samples (after cleaning)  
**Type:** Supervised regression (tabular)

## Feature Definitions (8 Inputs)

| Feature | Symbol | Physical Meaning | Typical Range | Units |
|---------|--------|------------------|---------------|-------|
| **M** | M | Magnetic field parameter | 0.0 - 2.0 | dimensionless |
| **S** | S | Rotation/angle parameter | 30° - 60° | ⚠️ **radians** |
| **K** | K | Permeability parameter | 0.5 - 2.0 | dimensionless |
| **phi1** | φ₁ | Nanoparticle volume fraction (type 1) | 0.00 - 0.05 | dimensionless |
| **phi2** | φ₂ | Nanoparticle volume fraction (type 2) | 0.00 - 0.02 | dimensionless |
| **Ec** | Ec | Eckert number (viscous dissipation) | 0.0 - 1.0 | dimensionless |
| **Pr** | Pr | Prandtl number (thermal diffusivity) | 50 - 300 | dimensionless |
| **eta** | η | Similarity variable (spatial) | 0.0 - 5.0 | dimensionless |

### Critical Unit Convention
⚠️ **ALERT:** The rotation parameter `S` **MUST** be in radians for solver consistency.  
**Detection rule:** If `S > 3.5`, assume degrees and convert: `S_rad = S_deg × π/180`

## Target Variables (2 Outputs)

| Target | Symbol | Physical Meaning | Typical Range | Interpretation |
|--------|--------|------------------|---------------|----------------|
| **f3** | f′′(η) | Velocity gradient | -2 to +2 | Shear stress at wall |
| **f5** | θ′(η) | Temperature gradient | -5 to +5 | Heat flux at wall |

### Physical Significance
- **f3 > 0:** Accelerating flow (stretching)
- **f3 < 0:** Decelerating flow (adverse pressure gradient)
- **f5 < 0:** Cooling (heat removal from sheet)
- **f5 > 0:** Heating (heat addition to sheet)

## Data Quality Requirements

The agent **MUST** enforce:

✅ **Completeness:** No missing values in f3 or f5  
✅ **Stability:** |f3|, |f5| < 10⁴ (solver blow-up threshold)  
✅ **Domain:** 0 ≤ η ≤ 5 (boundary layer extent)  
✅ **Units:** S in radians (convert if S > 3.5)  
✅ **Ordering:** `[M, S, K, phi1, phi2, Ec, Pr, eta, f3, f5]`  
✅ **Datatypes:** All float64 (no objects/strings)

---

# 🧹 4. DATA CLEANING RULES (MANDATORY)

These rules are **non-negotiable** and enforce physics-based data hygiene.

## Rule 1: Remove Missing Targets ❌

```python
# Drop rows where f3 or f5 is NaN
df = df.dropna(subset=['f3', 'f5'])
```

**Rationale:** Incomplete solver outputs indicate non-convergence or data corruption.

---

## Rule 2: Remove Solver Blow-Ups 💥

```python
# Remove numerical instabilities
df = df[(df['f3'].abs() < 1e4) & (df['f5'].abs() < 1e4)]
```

**Rationale:** |f| > 10⁴ indicates RK4 divergence, typically at:
- Extreme parameter combinations
- Near-singular points in the ODE system
- Numerical round-off accumulation

---

## Rule 3: Unit Conversion (Degrees → Radians) 🔄

```python
# Auto-detect and convert S
df.loc[df['S'] > 3.5, 'S'] = np.radians(df.loc[df['S'] > 3.5, 'S'])
```

**Rationale:** 
- Trigonometric terms in ODEs require radians
- S > 3.5 radians (~200°) is physically unrealistic for boundary layers
- Threshold 3.5 safely separates degrees (30-60) from radians (0.5-1.0)

---

## Rule 4: Domain Validation 📏

```python
# Enforce boundary-layer extent
df = df[(df['eta'] >= 0) & (df['eta'] <= 5)]
```

**Rationale:** 
- η = 0: wall surface
- η → ∞: free stream (practically η ≈ 5)
- Values outside indicate extrapolation beyond valid physics

---

## Rule 5: Column Ordering 🔢

```python
# Canonical order for reproducibility
COLUMNS = ['M', 'S', 'K', 'phi1', 'phi2', 'Ec', 'Pr', 'eta', 'f3', 'f5']
df = df[COLUMNS]
```

**Rationale:** Consistent ordering prevents input mismatch in trained models.

---

## Rule 6: Dataset Persistence 💾

```python
# Save cleaned master dataset
df.to_csv('clean_dataset.csv', index=False)

# Generate stratified split (80/20)
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2, random_state=42)

train.to_csv('data/processed/train_dataset.csv', index=False)
test.to_csv('data/processed/test_dataset.csv', index=False)
```

**Rationale:** 
- Reproducible splits via fixed random seed
- Separation of concerns (training vs evaluation)
- Audit trail for all data transformations

---

## Rule 7: Leakage Prevention 🔒

```python
# Verify no identical samples in train/test
train_tuples = set(map(tuple, train.values))
test_tuples = set(map(tuple, test.values))
assert len(train_tuples & test_tuples) == 0, "Data leakage detected!"
```

**Rationale:** Identical samples inflate test metrics artificially.

---

# 🤖 5. ML TASK DEFINITION

## Task A: Data Preprocessing Pipeline

### Deliverable: `preprocess.py`

**Functionality:**
```python
def preprocess_data(input_path='survey_sample_data.xlsx'):
    """
    Load raw Excel → Apply cleaning rules → Generate splits
    
    Returns:
        clean_dataset.csv
        data/processed/train_dataset.csv (80%)
        data/processed/test_dataset.csv (20%)
    """
```

**Steps:**
1. Load Excel (manual parsing, no external libs)
2. Apply Rules 1-5 (see Section 4)
3. Stratified split (preserve target distribution)
4. Export to CSV

---

## Task B: Classical ML Baselines

### Deliverable: `train_model.py --model-type classical`

**Algorithms to Compare:**
- Random Forest Regressor
- Gradient Boosting Regressor
- K-Nearest Neighbors
- **XGBoost** (recommended)
- **LightGBM** (recommended)

**Pipeline:**
```python
Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', MultiOutputRegressor(XGBRegressor()))
])
```

**Hyperparameters:**
```python
{
    'n_estimators': 600,
    'max_depth': 5,
    'learning_rate': 0.1,
    'tree_method': 'hist',
    'random_state': 42
}
```

**Output:** `models/classical_baseline.pkl`

---

## Task C: Neural Network Model

### Deliverable: `train_model.py --model-type neural`

**Architecture (PyTorch):**
```python
class NanofluidNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)  # No activation (regression)
```

**Training Configuration:**
```python
{
    'loss': nn.MSELoss(),
    'optimizer': torch.optim.Adam(lr=0.001),
    'batch_size': 32,
    'epochs': 500,
    'early_stopping': {'patience': 40, 'min_delta': 1e-4},
    'validation_split': 0.2
}
```

**Normalization:**
```python
# Z-score standardization (store in metadata)
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
X_train_norm = (X_train - X_mean) / X_std
```

**Outputs:**
- `models/neural_network.pt` (state dict)
- `models/neural_network_metadata.json` (means, stds, metrics)

---

## Task D: Unified Inference Interface

### Deliverable: `model_inference.py`

**Usage:**
```bash
# Single sample
python model_inference.py --model-type classical \
    --values 0.5 60 1 0.02 0.01 0.5 204 0.333

# Batch prediction
python model_inference.py --model-type neural \
    --input-csv samples.csv --output-csv predictions.csv
```

**Functionality:**
```python
def predict(model_type, input_params):
    """
    Auto-converts S if > 3.5
    Loads appropriate model (classical or neural)
    Returns: {'f3': value, 'f5': value}
    """
```

---

## Task E: Visualization Suite

### Deliverable: `visualize.py`

**Generate:**
1. **Parity plots** (predicted vs actual)
2. **Residual plots** (error vs predicted)
3. **Feature distributions** (train vs test)
4. **Correlation matrix**
5. **Cross-validation summary**
6. **Learning curves** (neural network)

**Output:** All plots saved to `reports/` directory

---

# 🧪 6. AGENT BEHAVIOR RULES

## Mandatory Behaviors ✅

### 1. Determinism
```python
# Fix all random seeds
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
```

**Rationale:** Scientific reproducibility requires identical results across runs.

### 2. Physics-Grounded Explanations

**Bad:** "The model learns patterns in the data."  
**Good:** "The model captures how magnetic parameter M suppresses velocity gradients f3 via Lorentz damping, consistent with the momentum equation."

**Requirement:** Every ML decision justified by physical principles.

### 3. Transparency

**Document:**
- Feature preprocessing steps
- Model architecture choices
- Hyperparameter rationale
- Validation methodology
- Known limitations

### 4. Error Handling

```python
# Graceful degradation
try:
    S_rad = convert_to_radians(S)
except ValueError:
    logger.error(f"Invalid S value: {S}")
    raise PhysicsViolationError("S outside valid range")
```

### 5. Unit Awareness

Always display units in outputs:
```
Predicted f3 = 0.543 (dimensionless velocity gradient)
Predicted f5 = -1.234 (dimensionless temperature gradient)
```

---

## Forbidden Actions ❌

### 1. Physics Fabrication

**Never:**
- Invent new governing equations
- Modify boundary conditions arbitrarily
- Create "synthetic" physics parameters

### 2. Feature Engineering Without Justification

**Prohibited:** Adding `f3 × f5` without dimensional analysis  
**Allowed:** Adding `M²K` (combined electromagnetic-permeability effect) with citation

### 3. Black-Box Claims

**Prohibited:** "The neural network learned complex relationships."  
**Required:** "The network's first hidden layer likely encodes Reynolds analogy (velocity-temperature coupling) as evidenced by activation clustering."

### 4. Ignoring Data Cleaning Rules

**All 7 rules (Section 4) are mandatory.** No exceptions.

### 5. Silent Failures

**Always log and raise exceptions for:**
- Missing files
- Invalid parameter ranges
- Convergence failures
- Data leakage detection
- Unit inconsistencies

---

## Decision Framework 🎯

When faced with choices, the agent must:

1. **Check FPF compliance** — Does this violate fundamental physics?
2. **Evaluate physical interpretability** — Can we explain this to a fluid dynamicist?
3. **Assess numerical stability** — Will this cause solver-like blow-ups?
4. **Document rationale** — Write clear justification in code comments
5. **Validate empirically** — Compare results against solver ground truth

---

# 📁 7. DELIVERABLES CHECKLIST

## Core Scripts ✅
- [x] `preprocess.py` — Data cleaning pipeline
- [x] `train_model.py` — Training entrypoint (classical + neural)
- [x] `model_inference.py` — Unified prediction interface
- [x] `visualize.py` — Diagnostic plot generator

## Datasets ✅
- [x] `clean_dataset.csv` — Cleaned master dataset
- [x] `data/processed/train_dataset.csv` — Training split
- [x] `data/processed/test_dataset.csv` — Test split

## Models ✅
- [x] `models/classical_baseline.pkl` — Best XGBoost pipeline
- [x] `models/neural_network.pt` — PyTorch state dict
- [x] `models/neural_network_metadata.json` — Normalization parameters

## Notebooks ✅
- [x] `notebooks/01_data_cleaning.ipynb` — Interactive preprocessing
- [x] `notebooks/02_classical_models.ipynb` — ML baseline comparison
- [x] `notebooks/03_neural_network.ipynb` — Neural network experiments

## Reports ✅
- [x] `reports/verification_summary.json` — Comprehensive metrics
- [x] `reports/classical_parity_*.png` — Parity plots
- [x] `reports/classical_residuals_*.png` — Residual diagnostics
- [x] `reports/neural_*.png` — Neural network diagnostics
- [x] `reports/correlation_matrix.png` — Feature analysis
- [x] `reports/feature_distributions.png` — Distribution comparison
- [x] `reports/cross_validation_summary.png` — CV results

## Documentation ✅
- [x] `README.md` — User-facing documentation
- [x] `AGENT.md` — This specification
- [x] `report.md` — Technical analysis report

---

# 📊 8. SUCCESS CRITERIA

## Model Performance Thresholds

### Minimum Acceptable Performance (MAP)
| Metric | f3 (velocity) | f5 (temperature) |
|--------|---------------|------------------|
| R² | ≥ 0.70 | ≥ 0.85 |
| RMSE | ≤ 0.25 | ≤ 1.00 |
| Max Error | ≤ 1.00 | ≤ 3.00 |

**Current Status:** ✅ XGBoost exceeds all MAP thresholds

### Data Quality Gates
- [x] Zero data leakage (verified)
- [x] All S values in radians (verified)
- [x] No solver blow-ups in cleaned data (verified)
- [x] Train/test distribution similarity (mild shifts documented)

### Code Quality Standards
- [x] PEP 8 compliance
- [x] Comprehensive docstrings
- [x] Exception handling
- [x] Logging infrastructure
- [x] Reproducible random seeds

---

# 🔄 9. MAINTENANCE PROTOCOL

## When to Retrain Models

**Triggers:**
1. New solver data available (>20% increase in dataset size)
2. Parameter range expansion (new M, S, K combinations)
3. Model performance degradation (R² drops >5%)
4. Physics equation updates (rare but critical)

**Procedure:**
```bash
# 1. Update raw data
cp new_solver_data.xlsx survey_sample_data.xlsx

# 2. Re-run preprocessing
python preprocess.py

# 3. Retrain models
python train_model.py --model-type classical --estimator xgboost
python train_model.py --model-type neural

# 4. Regenerate reports
python visualize.py

# 5. Validate against MAP thresholds
python -c "from reports.verification_summary import check_thresholds; check_thresholds()"
```

## Version Control Strategy

**Semantic Versioning:**
- **Major (X.0.0):** Physics equation changes
- **Minor (0.X.0):** New features or model architectures
- **Patch (0.0.X):** Bug fixes or documentation updates

**Git Tags:**
```bash
git tag -a v1.2.0 -m "Added LightGBM baseline, improved f5 RMSE by 12%"
```

---

# 🎓 10. KNOWLEDGE TRANSFER

## For New Developers

### Onboarding Checklist
1. Read `README.md` (user perspective)
2. Read this `AGENT.md` (system design)
3. Read `report.md` (analysis findings)
4. Run `01_data_cleaning.ipynb` (understand data pipeline)
5. Run `02_classical_models.ipynb` (understand baselines)
6. Experiment with inference: `python model_inference.py --help`

### Key Concepts to Master
- Dimensionless analysis (Buckingham π theorem)
- Boundary-layer theory (Blasius solution)
- Shooting method mechanics
- Multi-output regression vs separate models
- Physics-informed loss functions (future work)

### Common Pitfalls
⚠️ **Forgetting S unit conversion** → Always check S > 3.5  
⚠️ **Overfitting on small test set** → Use cross-validation  
⚠️ **Ignoring f5 bias** → Check residual plots religiously  
⚠️ **Breaking reproducibility** → Never modify random seeds  

---

# 📞 11. SUPPORT & ESCALATION

## Self-Service Resources
1. Check `reports/verification_summary.json` for metrics
2. Review residual plots for systematic errors
3. Consult notebooks for interactive debugging
4. Search GitHub Issues for similar problems

## Escalation Path
1. **Level 1:** Documentation / README FAQ
2. **Level 2:** GitHub Issues (technical questions)
3. **Level 3:** Direct maintainer contact (critical physics questions)

---

# 🔚 END OF AGENT SPECIFICATION

**Document Version:** 2.0  
**Last Updated:** 2025-11-13  
**Status:** Active (production-ready models deployed)

**Next Review:** Upon dataset expansion or physics model update

---

*This specification is a living document. All modifications must preserve FPF compliance and maintain backward compatibility with existing trained models.*

