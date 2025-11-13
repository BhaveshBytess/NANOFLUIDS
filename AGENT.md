# 🧠 **AGENT SPECIFICATION — Hybrid Nanofluid ML Project**

## **📌 Project Name:**

Hybrid Nanofluid Boundary-Layer ML Predictor

## **📌 Project Goal**

Train a machine learning model that predicts:

* **f3(η) = f′′(η)** → velocity gradient
* **f5(η) = θ′(η)** → temperature gradient

from physical input parameters of a hybrid nanofluid flow system.

---

# 🚀 **1. FIRST PRINCIPLES FRAMEWORK (FPF)**

The agent must follow FPF at all times:

### **FPF Rule 1 – No hallucinations**

All claims must trace back to physics, equations, or dataset values.

### **FPF Rule 2 – Explain from fundamentals**

Always connect every ML decision to physical meaning.

### **FPF Rule 3 – Preserve equations**

Never modify governing ODEs, boundary conditions, or physical constants on its own.

### **FPF Rule 4 – Respect numerical solver behavior**

Recognize that f3 and f5 come from RK4 + shooting method.
The ML model must mimic the solver, not replace physics.

---

# 🌡 **2. PHYSICS BACKGROUND**

The project is based on boundary-layer flow of hybrid nanofluid over a stretching sheet.

The MATLAB solver computes:

* **f1** = f
* **f2** = f′
* **f3** = f′′ (velocity gradient → output to learn)
* **f4** = θ
* **f5** = θ′ (temperature gradient → output to learn)

These come from solving five ODEs using:

* Runge–Kutta 4
* Shooting method to adjust s01, s02, s11, s12

---

# 📊 **3. DATASET STRUCTURE**

Your ONE MASTER DATASET contains:

## **INPUTS (8 features)**

| Feature | Meaning                                    |
| ------- | ------------------------------------------ |
| M       | Magnetic field parameter                   |
| S       | Rotation/angle parameter (MUST be radians) |
| K       | Permeability parameter                     |
| phi1    | Nanoparticle volume fraction #1            |
| phi2    | Nanoparticle volume fraction #2            |
| Ec      | Eckert number                              |
| Pr      | Prandtl number                             |
| eta     | Similarity variable (0 → ~5)               |

## **OUTPUTS (2 labels)**

| Output | Meaning                      |
| ------ | ---------------------------- |
| f3     | f′′(η) → Velocity gradient   |
| f5     | θ′(η) → Temperature gradient |

The agent must:

* Ensure input/output columns exist
* Verify correct datatypes
* Verify S is in radians, not degrees
* Validate η ranges
* Detect unstable solver outputs
* Remove ≥10⁴ blow-up values

---

# 🧹 **4. DATA CLEANING RULES (MANDATORY)**

The agent must apply:

### ✔ Rule 1 — Remove rows with missing f3 or f5

### ✔ Rule 2 — Remove rows where |f3| or |f5| > 10,000

(indicates RK4 divergence)

### ✔ Rule 3 — Convert all S (degrees → radians) if S > 3.5

Because sin(S)^2 expects radians.

### ✔ Rule 4 — Normalize inputs (optional)

### ✔ Rule 5 — Restore correct column ordering

### ✔ Rule 6 — Save final dataset as `clean_dataset.csv`

---

# 🤖 **5. ML TASK DEFINITION**

The Agent must produce:

### **Task A — Data preprocessing script**

* Load CSV
* Clean
* Convert S
* Split into train/test

### **Task B — Neural network model**

A multi-output regression model:

**Inputs:** 8 features
**Outputs:** 2 values → [f3, f5]

Recommended architecture:

```
Dense(64, relu)
Dense(128, relu)
Dense(64, relu)
Dense(2, linear)
```

Framework:

* PyTorch **or** TensorFlow (developer decides)

### **Task C — Training loop**

* MSE loss (multi-output)
* Adam optimizer
* Early stopping
* Validation curve plots

### **Task D — Inference script**

Given:

```
M, S, K, phi1, phi2, Ec, Pr, eta
→ return f3, f5
```

---

# 🧪 **6. AGENT BEHAVIOR RULES**

### The agent must:

* Be deterministic
* Not fabricate physics values
* Not add features unless requested
* Always explain ML decisions with physical reasoning
* Follow FPF rigorously

### The agent must NOT:

* Invent equations
* Modify the numerical solver logic
* Change columns
* Ignore cleaning rules
* Replace scientific meaning with generic ML talk

---

# 📁 **7. FILES THE AGENT MUST GENERATE**

1. `clean_dataset.csv`
2. `train_model.py`
3. `preprocess.py`
4. `model_inference.py`
5. `README.md` (auto-generated documentation)

---

# 🔚 **END OF agent.md**

