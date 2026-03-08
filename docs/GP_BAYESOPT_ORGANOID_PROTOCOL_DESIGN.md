---
date: 2026-03-03
title: Gaussian Processes and Bayesian Optimization for Brain Organoid Protocol Design
tags: [bayesian-optimization, gaussian-process, organoid, morphogen, active-learning, machine-learning]
---

# Gaussian Processes and Bayesian Optimization for Brain Organoid Protocol Design with <200 Training Datapoints

> **Status**: Implementation-ready research reference. Covers GPerturb, BoTorch MOBO, scikit-learn GP, and active learning loop design. Directly applicable to EC (endothelial cell) optimization from the Sanchis-Calleja 97-condition morphogen screen dataset.

---

## 1. GPerturb (Xing & Yau, Nature Communications 2025)

### Publication and Repository

- **Paper**: Xing H, Yau C. "GPerturb: Gaussian process modelling of single-cell perturbation data." *Nature Communications* 16, 5423 (2025). https://doi.org/10.1038/s41467-025-61165-7
- **GitHub**: https://github.com/hwxing3259/GPerturb (MIT license, Python ≥3.8)
- **bioRxiv preprint**: https://www.biorxiv.org/content/10.1101/2025.03.26.645455v1.full
- **Install**: `pip install git+https://github.com/hwxing3259/GPerturb.git`

### What GPerturb Does

GPerturb is a Bayesian sparse additive perturbation regression model for single-cell data. Its core innovation is decomposing observed gene expression into two additive GP components:

```
y_i = f_basal(c_i) + f_perturbation(p_i) + noise_i
```

Where:
- `y_i` = observed gene expression for cell i (G-dimensional)
- `c_i` = cell-level covariates (cell type identity, sequencing depth, etc.)
- `p_i` = perturbation descriptor vector (D-dimensional: which morphogens, at what doses)
- `f_basal(·)` = GP modeling unperturbed background expression for cell type
- `f_perturbation(·)` = sparse GP modeling additive perturbation effect
- Sparsity: a binary on/off switch per gene controls whether a perturbation affects it at all

The key design decision is that the perturbation component is **sparse and gene-specific**: most genes will not respond to a given morphogen, and GPerturb infers which ones do via a binary indicator prior. This directly produces interpretable uncertainty estimates — not just "what is the effect" but "how confident are we that any effect exists."

### Two Observation Models

- **GPerturb-Gaussian**: for log-transformed, normalized expression values (standard scRNA-seq pipeline output)
- **GPerturb-ZIP**: for raw UMI counts (zero-inflated Poisson likelihood)

For morphogen dose-response data from a plate-based screen (97 conditions × cells), GPerturb-Gaussian is the appropriate choice after log-normalizing counts.

### API and Data Format

GPerturb requires three matrices as input:

| Matrix | Shape | Content |
|--------|-------|---------|
| `observation` | N_cells × G_genes | Normalized expression (log1p or similar) |
| `cell_info` | N_cells × K | Cell covariates (log_counts, gene_density, etc.) |
| `perturbation` | N_cells × D | Morphogen perturbation descriptor for each cell |

The perturbation matrix `P` is the key encoding. For morphogen data, each column is a morphogen (CHIR, BMP4, SHH, RA, FGF8, etc.) and each row is a cell. The value is the concentration the cell was exposed to. **Important**: the authors apply a power transform to dosages before input: `P_transformed = P^0.2`. This compresses the dose range and prevents high-concentration conditions from dominating kernel distances.

```python
import torch
import numpy as np
from GPerturb import GPerturb_model

# --- Prepare data from 97-condition Sanchis-Calleja screen ---
# observation: (N_cells, G_genes) float tensor, log-normalized counts
# cell_info:   (N_cells, K) float tensor — [log_counts, gene_density, ...]
# perturbation: (N_cells, D) float tensor — morphogen concentrations

# Power-transform dosage (critical preprocessing step from paper)
perturbation_transformed = torch.pow(perturbation, 0.2)

# Instantiate model
model = GPerturb_model.GPerturb_Gaussian(
    conditioner_dim=perturbation_transformed.shape[1],   # D: number of morphogens
    output_dim=observation.shape[1],                      # G: number of genes
    base_dim=cell_info.shape[1],                          # K: cell covariates
    data_size=observation.shape[0],                       # N: total cells
    hidden_node=700,
    hidden_layer_1=2,
    hidden_layer_2=2,
    tau=0.01   # sparsity hyperparameter — tighter = fewer genes affected
)

# Split test set (done before training)
model.test_id = np.random.choice(N_cells, size=int(0.2 * N_cells), replace=False)

# Train
lr_parametric = 0.001
model.GPerturb_train(
    epoch=250,
    observation=observation,
    cell_info=cell_info,
    perturbation=perturbation_transformed,
    lr=lr_parametric,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)
```

**Runtime note**: The SciPlex2 demo (large dataset) takes ~1.5 hours on a desktop with RTX 2060. For a modest 97-condition screen with ~100K cells total (~1K cells/condition), expect 15–30 minutes on a CPU-only laptop with reduced `hidden_node=300` and `epoch=100`.

### Adapting GPerturb for Morphogen Dose-Response

The standard GPerturb use case is discrete CRISPR perturbations (gene X knocked out vs. not). Adapting to morphogen dose-response requires treating dose as a **continuous perturbation variable**, not a binary indicator:

1. **Perturbation matrix encoding**: Use continuous concentration values (µM or ng/mL), power-transformed. A condition with CHIR 3µM and BMP4 10 ng/mL + no SHH becomes row `[3.0^0.2, 10.0^0.2, 0.0^0.2, ...]` = `[1.246, 1.585, 0.0, ...]`.

2. **Zero concentration handling**: Untreated wells have `P[i,j] = 0` for all morphogens. This is the control/basal condition — GPerturb's basal component `f_basal` should capture this.

3. **Extracting the perturbation effect for a novel dose**: After training, to predict gene expression under a new unseen condition (e.g., CHIR 0.75µM, BMP4 5 ng/mL), create a synthetic cell row with the corresponding power-transformed concentrations and query the model.

4. **What GPerturb gives you for active learning**: The sparse binary indicator posterior tells you which genes respond to a morphogen. The per-gene uncertainty estimates tell you which conditions are most uncertain. Use conditions with high posterior variance as candidates for the next experimental round.

### Critical Limitation of GPerturb for This Use Case

GPerturb is primarily a **retrospective analysis tool** — it fits on observed scRNA-seq data to characterize what happened. It does not natively implement an acquisition function or recommend the next experiment. To close the active learning loop, you need to combine GPerturb with a Bayesian Optimization layer (see Section 2) that takes GPerturb's uncertainty estimates and proposes new conditions.

---

## 2. Bayesian Optimization for Next-Experiment Design

### 2.1 Encoding Morphogen Conditions as Input Vectors

Each experimental condition is a point in a D-dimensional continuous input space where D = number of morphogens × number of time windows.

For the Sanchis-Calleja dataset covering CHIR, SHH, FGF8, RA, BMP4, BMP7, XAV939, purmorphamine across early/mid/late windows, a minimal encoding for a CHIR × BMP4 two-factor screen is:

```python
# Condition encoding: x = [CHIR_conc, BMP4_conc, timing_window]
# timing_window: 0=days0-3, 1=days0-9, 2=days3-9 (encoded as 0/1/2 or one-hot)

# For a 6x4 CHIR x BMP4 dose matrix (24 conditions):
CHIR_doses  = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]   # µM
BMP4_doses  = [0.0, 5.0, 10.0, 25.0]             # ng/mL

conditions = []
for chir in CHIR_doses:
    for bmp4 in BMP4_doses:
        conditions.append([chir, bmp4, 0])  # timing window = days 0-3

X_candidates = torch.tensor(conditions, dtype=torch.double)  # shape: (24, 3)
```

For a richer multi-morphogen, multi-timing design:

```python
# Full 8-morphogen encoding with log-transform for wide concentration ranges
morphogens = ["CHIR", "BMP4", "SHH", "RA", "FGF8", "BMP7", "XAV939", "purmo"]
# Concentration + binary timing flags: day0-3, day3-9, day9-21
# x_i = [log(conc_CHIR+1), log(conc_BMP4+1), ..., t0, t1, t2]
# Total input dim D = 8 morphogens + 3 timing windows = 11
```

**Key preprocessing decisions**:
- Log-transform concentrations (`log(c + epsilon)`) if ranges span orders of magnitude (0.1 µM to 100 µM)
- Normalize all inputs to [0, 1] using known biological ranges before fitting
- Do NOT normalize outputs (cell type fraction) — keep as raw proportions for interpretability

### 2.2 Kernel Selection: Matérn vs. RBF and Additive Kernels

#### Why Matérn over RBF

The RBF (squared exponential) kernel assumes **infinitely differentiable** response surfaces. For biological dose-response data, this is almost certainly wrong: there are often threshold effects, saturation at high doses, and non-monotonic behaviors that produce functions with finite smoothness. The **Matérn-5/2 kernel** is the standard choice for biological optimization because:

- Matérn-5/2 assumes functions are twice-differentiable — biologically plausible for smooth dose-response curves
- Matérn-3/2 is appropriate if you expect sharper transitions (e.g., binary fate decisions at a threshold concentration)
- RBF will over-smooth and fail to capture biological non-linearities

```python
# Matérn 5/2 kernel formula:
# k(r) = (1 + sqrt(5)*r/l + 5*r^2/(3*l^2)) * exp(-sqrt(5)*r/l)
# where r = ||x - x'||_2, l = lengthscale

from gpytorch.kernels import MaternKernel, ScaleKernel
kernel = ScaleKernel(MaternKernel(nu=2.5))  # nu=2.5 => Matern-5/2
```

#### Additive Kernels for Combination Effects

When morphogen A and morphogen B both influence a readout, the total effect may be:
- **Purely additive**: EC fraction = f(CHIR) + g(BMP4) — no interaction
- **Synergistic**: EC fraction > f(CHIR) + g(BMP4)
- **Antagonistic**: EC fraction < f(CHIR) + g(BMP4)

A **sum + product additive kernel** with Matérn base kernels captures all three cases:

```python
import gpytorch
import torch

class AdditiveMaternGP(gpytorch.models.ExactGP):
    """
    Additive GP with pairwise interaction terms.
    Captures individual morphogen effects + synergistic/antagonistic combinations.
    Scales as O(D^2) in kernel computation — tractable for D ≤ 10 morphogens.
    """
    def __init__(self, train_x, train_y, D, nu=2.5, max_degree=2):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # D independent Matern kernels (one per morphogen), batched
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=nu,
                batch_shape=torch.Size([D]),
                ard_num_dims=1   # one lengthscale per morphogen dimension
            )
        )
        self.max_degree = max_degree  # 2 = include pairwise interactions

    def forward(self, x):
        mean_x = self.mean_module(x)
        # Reshape: (D, N, 1) — one kernel per morphogen
        batched = x.mT.unsqueeze(-1)
        univariate_covars = self.covar_module(batched)
        # sum_interaction_terms computes: Σ k_i + Σ k_i*k_j + ... in O(D^2)
        covar = gpytorch.utils.sum_interaction_terms(
            univariate_covars, max_degree=self.max_degree, dim=-3
        )
        return gpytorch.distributions.MultivariateNormal(mean_x, covar)
```

**Kernel recommendation table**:

| Scenario | Kernel | Rationale |
|----------|--------|-----------|
| Single morphogen dose-response | `ScaleKernel(MaternKernel(nu=2.5))` | Standard for smooth 1D biological curves |
| 2–3 morphogen combination, additive effects expected | `AdditiveMaternGP(max_degree=1)` | Main effects only, faster |
| 2–3 morphogen combo, synergy possible | `AdditiveMaternGP(max_degree=2)` | Captures pairwise interactions |
| Full 8-morphogen screen | `ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=8))` | ARD lengthscales identify irrelevant morphogens |
| Large library with known inactive morphogens | `ScaleKernel(MaternKernel) + WhiteKernel` | Add WhiteKernel for observation noise |

### 2.3 Acquisition Function Choice

#### Expected Improvement (EI)

EI measures the expected gain over the current best observation:

```
EI(x) = E[max(f(x) - f_best, 0)]
       = (µ(x) - f_best) * Φ(Z) + σ(x) * φ(Z)
       where Z = (µ(x) - f_best) / σ(x)
```

- **Recommended for**: Optimization problems where you want to find the single best condition as efficiently as possible
- **Behavior**: Naturally balances exploration (high σ) and exploitation (high µ)
- **Problem**: Standard EI has flat gradients away from the current best, making optimization numerically difficult. Use **LogEI** in BoTorch, which avoids this.

```python
from botorch.acquisition import LogExpectedImprovement
logEI = LogExpectedImprovement(model=gp_model, best_f=train_Y.max())
```

#### Upper Confidence Bound (UCB)

```
UCB(x) = µ(x) + β * σ(x)
```

- **Recommended for**: Hackathon/rapid prototyping use — the `β` parameter gives explicit control over exploration-exploitation trade-off
- **β = 0.1**: Heavy exploitation, recommends near-current-best conditions
- **β = 2.0**: Balanced exploration/exploitation (standard choice)
- **β = 5.0**: Aggressive exploration, samples uncertain regions

```python
from botorch.acquisition import UpperConfidenceBound
UCB = UpperConfidenceBound(model=gp_model, beta=2.0)
```

**For a hackathon starting from scratch with 97 training points**: Use **UCB with β=2.0** for Round 1 because:
1. You have zero sequencing data from your own conditions — explore broadly
2. UCB has no numerical instability issues at the boundaries of the training data
3. Easy to tune β up/down based on intuition

**Switch to LogEI for Round 2+** once you have your own data and want to converge on the best conditions.

#### For Multi-Objective: qLogNEHVI

When optimizing simultaneously for EC fraction (maximize) and total off-target cell types (minimize), use the Expected Hypervolume Improvement acquisition function:

```python
from botorch.acquisition.multi_objective import qLogNoisyExpectedHypervolumeImprovement

# Objective 1: EC fraction (maximize)
# Objective 2: -total_off_target_fraction (maximize negative = minimize off-targets)
acq_func = qLogNoisyExpectedHypervolumeImprovement(
    model=multi_output_gp,
    ref_point=torch.tensor([-0.1, -1.0]),  # slightly worse than worst observed
    X_baseline=train_X,
    prune_baseline=True,   # keep only Pareto-efficient baseline points
    sample_shape=torch.Size([128]),  # MC samples for integration
)
```

### 2.4 BoTorch Implementation Details

#### Minimum Working Example: Full BO Loop

```python
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf

# IMPORTANT: Use double precision throughout to avoid numerical issues
torch.set_default_dtype(torch.float64)

def run_bo_round(train_X, train_Y, bounds, n_candidates=1):
    """
    Single BO round: fit GP, compute acquisition function, return next conditions.

    Args:
        train_X: (N, D) tensor of normalized morphogen conditions
        train_Y: (N, 1) tensor of EC fraction measurements (0-1)
        bounds: (2, D) tensor with lower/upper bounds for each morphogen dim
        n_candidates: how many new conditions to recommend (batch size)

    Returns:
        candidates: (n_candidates, D) tensor of recommended next conditions
    """
    # --- Step 1: Fit GP ---
    gp = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    gp.eval()

    # --- Step 2: Define acquisition function ---
    logEI = LogExpectedImprovement(model=gp, best_f=train_Y.max())

    # --- Step 3: Optimize acquisition function over the feasible space ---
    candidates, acq_value = optimize_acqf(
        acq_function=logEI,
        bounds=bounds,
        q=n_candidates,           # batch size
        num_restarts=10,          # number of random starts for gradient optimization
        raw_samples=128,          # sobol samples for initialization
        options={"maxiter": 200},
    )

    return candidates.detach()

# --- Set up bounds for a CHIR x BMP4 screen ---
# Input: x = [CHIR (µM), BMP4 (ng/mL)] normalized to [0,1]
# CHIR range: 0 - 5 µM; BMP4 range: 0 - 25 ng/mL
bounds = torch.tensor([[0.0, 0.0],   # lower bounds
                        [1.0, 1.0]]) # upper bounds (1.0 = normalized max)

# Example: 97-condition training data
# train_X would be your normalized morphogen concentrations
# train_Y would be EC fraction from scRNA-seq deconvolution
next_conditions = run_bo_round(train_X, train_Y, bounds, n_candidates=24)

# De-normalize to get actual concentrations to run in lab:
CHIR_range = 5.0   # µM
BMP4_range = 25.0  # ng/mL
actual_conditions = next_conditions.clone()
actual_conditions[:, 0] *= CHIR_range
actual_conditions[:, 1] *= BMP4_range

print("Recommended CHIR × BMP4 conditions:")
for i, (chir, bmp4) in enumerate(actual_conditions):
    print(f"  Condition {i+1:02d}: CHIR {chir:.3f} µM, BMP4 {bmp4:.2f} ng/mL")
```

### 2.5 Multi-Objective Optimization: Maximize EC + Minimize Off-Targets

For a realistic organoid protocol optimization, you simultaneously want:
1. **High EC (endothelial cell) fraction** — direct target
2. **Low astrocyte contamination** — common off-target
3. **Low undifferentiated (pluripotent) cell fraction** — safety concern

This is a 3-objective Pareto optimization problem:

```python
import torch
from botorch.models import ModelListGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.multi_objective import qLogNoisyExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective import is_non_dominated
from gpytorch.mlls import SumMarginalLogLikelihood

def multi_objective_bo_round(train_X, train_Y_list, bounds, ref_point, batch_size=8):
    """
    Multi-objective BO round optimizing EC fraction + off-target minimization.

    Args:
        train_X: (N, D) morphogen conditions
        train_Y_list: list of (N,1) tensors: [EC_frac, -astrocyte_frac, -undiff_frac]
                     NOTE: objectives to maximize (negate to minimize)
        bounds: (2, D) feasibility bounds
        ref_point: (K,) reference point slightly below worst observed values
        batch_size: number of conditions to recommend

    Returns:
        candidates: (batch_size, D) Pareto-optimal candidate conditions
        pareto_X: current Pareto-optimal training conditions
    """
    # Stack objectives: (N, K) where K = number of objectives
    train_Y_stacked = torch.cat(train_Y_list, dim=-1)

    # Identify current Pareto front in training data
    pareto_mask = is_non_dominated(train_Y_stacked)
    pareto_X = train_X[pareto_mask]

    # Fit independent GPs for each objective (ModelListGP)
    models = [SingleTaskGP(train_X, y) for y in train_Y_list]
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()

    # Expected Hypervolume Improvement acquisition function
    acq = qLogNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,     # reference "worst acceptable" point
        X_baseline=train_X,
        prune_baseline=True,
        sample_shape=torch.Size([256]),
    )

    # Optimize acquisition function to get next batch
    candidates, _ = optimize_acqf(
        acq_function=acq,
        bounds=bounds,
        q=batch_size,
        num_restarts=5,
        raw_samples=64,
        sequential=True,  # sequential greedy for batch
    )

    return candidates.detach(), pareto_X

# Usage example:
# ref_point = (slightly worse than worst observed) for each objective
# For EC (maximize): ref = 0.0 (no EC is worst)
# For -astrocyte (maximize neg): ref = -1.0 (100% astrocyte is worst)
# For -undiff (maximize neg): ref = -1.0
ref_point = torch.tensor([0.0, -1.0, -1.0])

next_conditions, current_pareto = multi_objective_bo_round(
    train_X, [ec_frac, neg_astrocyte_frac, neg_undiff_frac],
    bounds, ref_point, batch_size=24
)
```

---

## 3. Practical Hackathon Implementation

### 3.1 Can This Run on a Laptop in <10 Minutes?

**Yes, for the scikit-learn path. Borderline for BoTorch.**

| Framework | N=97, D=2 | N=97, D=8 | N=200, D=8 |
|-----------|-----------|-----------|------------|
| scikit-learn GPR (fit + predict) | <1 second | ~2 seconds | ~5 seconds |
| scikit-learn GPR + manual EI loop | ~10 seconds | ~30 seconds | ~60 seconds |
| BoTorch SingleTaskGP (CPU) | ~5 seconds | ~15 seconds | ~30 seconds |
| BoTorch MOBO qLogNEHVI (CPU) | ~30 seconds | ~2 minutes | ~5 minutes |
| GPerturb full model (CPU) | 15–30 min | 30–60 min | >1 hour |

**Bottleneck analysis**:
- GP fitting (hyperparameter optimization via L-BFGS-B): O(N^3) Cholesky, but N=97 → 97^3 ≈ 912,673 operations. Negligible on any modern hardware.
- Acquisition function optimization: The expensive part is `optimize_acqf` with `num_restarts=10` and `raw_samples=128`. Each restart runs L-BFGS-B to convergence. For D=2, this is sub-second. For D=8 with 24 parallel candidates (q=24), budget ~2 minutes.
- GPerturb: Neural network training with 250 epochs is the expensive part. Cut to `epoch=50` for a 5x speedup at cost of some convergence.

**Hackathon recommendation**: Use scikit-learn for initial prototyping (Section 3.3), then port to BoTorch for the full active learning loop.

### 3.2 scikit-learn GP vs GPyTorch/BoTorch Trade-offs

| Criterion | scikit-learn `GaussianProcessRegressor` | GPyTorch + BoTorch |
|-----------|----------------------------------------|---------------------|
| Setup time | 5 lines of code | 30-50 lines of code |
| Kernel flexibility | Good (Matern, RBF, Dot, Sum, Product) | Excellent (any kernel, custom kernels) |
| Acquisition functions | Manual implementation required | Built-in: LogEI, UCB, qNEHVI, LogPI |
| Multi-objective optimization | Not supported | First-class support |
| Batch recommendations (q>1) | Manual | Built-in `q` parameter |
| Uncertainty calibration | Good for small data | Excellent, GPU-accelerated |
| GPU support | None | Yes (CUDA) |
| Learning curve | Minimal | Moderate (PyTorch familiarity needed) |
| When to use | Prototyping, single objective, <10 min | Production active learning loop |

**Key scikit-learn limitation**: It has no built-in acquisition function optimization. You must either (a) evaluate the acquisition function on a discrete grid of candidate conditions or (b) implement gradient-free optimization (Nelder-Mead, Powell) manually.

### 3.3 Minimum Code: Fit a GP on 97-Condition Morphogen Screen Data

This is a complete working script for Round 0 of the active learning loop.

```python
"""
gp_morphogen_round0.py
Fit a GP to Sanchis-Calleja 97-condition morphogen screen.
Predict EC fraction + uncertainty for novel CHIR x BMP4 conditions.
Runtime: ~30 seconds on any laptop. No GPU required.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 1. Load data (replace with actual Sanchis-Calleja parsed outputs)
# =============================================================================
# After running cell type deconvolution (e.g., with sctype or scAnnotatR)
# on the 97-condition scRNA-seq data, extract:
#   - X: (97, D) array of morphogen conditions
#   - y: (97,)   array of EC (endothelial cell) fraction per condition

# Example: loading from parsed CSV
# df = pd.read_csv("sanchis_calleja_conditions_with_EC_fraction.csv")
# X_raw = df[["CHIR_uM", "BMP4_ngmL", "SHH_ngmL", "RA_uM"]].values
# y = df["EC_fraction"].values

# For demonstration, simulate the data structure:
np.random.seed(42)
N_CONDITIONS = 97
D = 4  # CHIR, BMP4, SHH, RA (main morphogens from the screen)

# Simulate morphogen concentrations (replace with real data)
X_raw = np.zeros((N_CONDITIONS, D))
X_raw[:, 0] = np.random.choice([0, 1, 3, 5], N_CONDITIONS)       # CHIR µM
X_raw[:, 1] = np.random.choice([0, 5, 10, 25, 50], N_CONDITIONS)  # BMP4 ng/mL
X_raw[:, 2] = np.random.choice([0, 100, 500], N_CONDITIONS)       # SHH ng/mL
X_raw[:, 3] = np.random.choice([0, 1, 10], N_CONDITIONS)          # RA nM

# Simulated EC fractions (replace with deconvolved scRNA-seq outputs)
# In real data: y = scRNA-seq cell type deconvolution result
y = np.clip(
    0.05 * X_raw[:, 0]  +              # CHIR weakly promotes EC
    0.003 * X_raw[:, 1]  +             # BMP4 strongly promotes EC
    -0.0001 * X_raw[:, 2] +            # SHH suppresses EC
    np.random.normal(0, 0.03, N_CONDITIONS),
    0, 1
)

print(f"Loaded {N_CONDITIONS} conditions, {D} morphogens")
print(f"EC fraction: min={y.min():.3f}, max={y.max():.3f}, mean={y.mean():.3f}")

# =============================================================================
# 2. Preprocessing: log-transform concentrations + standardize
# =============================================================================
# Log transform: crucial for concentrations spanning multiple orders of magnitude
X_log = np.log1p(X_raw)  # log(c + 1) handles zeros naturally

# Standardize inputs to zero mean, unit variance
# This ensures kernel lengthscales are on comparable scales
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

# =============================================================================
# 3. Define and fit GP
# =============================================================================
# Kernel: ConstantKernel (output scale) * Matern-5/2 (smooth dose-response)
#       + WhiteKernel (observation noise)
kernel = (
    ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) *
    Matern(length_scale=np.ones(D), length_scale_bounds=(1e-2, 1e2), nu=2.5) +
    WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 1e-1))
)
# ARD: length_scale=np.ones(D) gives one lengthscale per morphogen.
# After fitting, short lengthscale => morphogen strongly influences EC fraction.
# Long lengthscale => morphogen weakly influences EC fraction.

gpr = GaussianProcessRegressor(
    kernel=kernel,
    alpha=0.0,          # noise handled by WhiteKernel — set alpha=0
    n_restarts_optimizer=10,  # 10 random restarts for kernel hyperparameter fit
    normalize_y=True,   # zero-center targets (important for GP prior)
    random_state=42
)

print("\nFitting GP on 97 training conditions...")
gpr.fit(X_scaled, y)
print(f"Fitted kernel: {gpr.kernel_}")

# =============================================================================
# 4. Predict EC fraction + uncertainty for novel conditions
# =============================================================================
# Query: What EC fraction does the GP predict for CHIR=0.75µM, BMP4=8 ng/mL?
# This is the "EC at CHIR 0.75 µM" example from the problem statement.

novel_conditions = np.array([
    [0.75, 8.0,   0.0, 0.0],   # CHIR 0.75 µM + BMP4 8 ng/mL
    [2.0,  10.0,  0.0, 0.0],   # CHIR 2 µM + BMP4 10 ng/mL
    [1.0,  0.0,  100.0, 0.0],  # CHIR 1 µM + SHH 100 ng/mL
    [3.0,  25.0, 500.0, 1.0],  # High everything
])

# Apply same preprocessing as training data
X_novel_log = np.log1p(novel_conditions)
X_novel_scaled = scaler.transform(X_novel_log)

# Predict mean + standard deviation (epistemic uncertainty)
y_pred, y_std = gpr.predict(X_novel_scaled, return_std=True)

print("\nPredictions for novel conditions:")
print(f"{'Condition':<45} {'EC fraction':>12} {'± 95% CI':>10}")
print("-" * 70)
labels = [
    "CHIR 0.75µM + BMP4 8 ng/mL",
    "CHIR 2µM + BMP4 10 ng/mL",
    "CHIR 1µM + SHH 100 ng/mL",
    "High all morphogens",
]
for label, mu, sigma in zip(labels, y_pred, y_std):
    ci95 = 1.96 * sigma  # 95% confidence interval
    print(f"{label:<45} {mu:>10.3f}   ± {ci95:.3f}")

# =============================================================================
# 5. Identify most uncertain conditions in a candidate grid
# =============================================================================
# For next experiment selection: scan a dense grid and find where uncertainty is highest
CHIR_grid = np.linspace(0, 5, 20)
BMP4_grid = np.linspace(0, 25, 20)

CHIRg, BMP4g = np.meshgrid(CHIR_grid, BMP4_grid)
X_grid_raw = np.column_stack([
    CHIRg.ravel(),
    BMP4g.ravel(),
    np.zeros(400),   # SHH = 0
    np.zeros(400)    # RA = 0
])
X_grid_log = np.log1p(X_grid_raw)
X_grid_scaled = scaler.transform(X_grid_log)

y_grid_pred, y_grid_std = gpr.predict(X_grid_scaled, return_std=True)

# Top 10 most uncertain conditions (candidates for next experiment)
top_uncertain_idx = np.argsort(y_grid_std)[-10:][::-1]
print("\nTop 10 most uncertain candidate conditions (explore these):")
print(f"{'CHIR (µM)':>12} {'BMP4 (ng/mL)':>14} {'Predicted EC':>13} {'Std Dev':>10}")
for idx in top_uncertain_idx:
    print(f"{X_grid_raw[idx, 0]:>12.2f} {X_grid_raw[idx, 1]:>14.2f} "
          f"{y_grid_pred[idx]:>13.3f} {y_grid_std[idx]:>10.3f}")

# Top 10 highest predicted EC fraction (exploit these)
top_predicted_idx = np.argsort(y_grid_pred)[-10:][::-1]
print("\nTop 10 highest predicted EC fraction (exploit these):")
print(f"{'CHIR (µM)':>12} {'BMP4 (ng/mL)':>14} {'Predicted EC':>13} {'Std Dev':>10}")
for idx in top_predicted_idx:
    print(f"{X_grid_raw[idx, 0]:>12.2f} {X_grid_raw[idx, 1]:>14.2f} "
          f"{y_grid_pred[idx]:>13.3f} {y_grid_std[idx]:>10.3f}")

# =============================================================================
# 6. Compute Expected Improvement manually (scikit-learn has no built-in EI)
# =============================================================================
from scipy.stats import norm

def expected_improvement(mu, sigma, best_f, xi=0.01):
    """
    Expected Improvement acquisition function.
    xi: exploration bonus (larger = more exploration). Typical range: 0.001 - 0.1
    """
    Z = (mu - best_f - xi) / (sigma + 1e-9)
    EI = (mu - best_f - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    EI[EI < 0] = 0
    return EI

best_observed = y.max()
EI_values = expected_improvement(y_grid_pred, y_grid_std, best_f=best_observed, xi=0.01)

# Top EI conditions (balance explore/exploit)
top_EI_idx = np.argsort(EI_values)[-10:][::-1]
print("\nTop 10 conditions by Expected Improvement (balanced explore/exploit):")
print(f"{'CHIR (µM)':>12} {'BMP4 (ng/mL)':>14} {'Predicted EC':>13} {'EI score':>10}")
for idx in top_EI_idx:
    print(f"{X_grid_raw[idx, 0]:>12.2f} {X_grid_raw[idx, 1]:>14.2f} "
          f"{y_grid_pred[idx]:>13.3f} {EI_values[idx]:>10.4f}")
```

**Expected runtime**: <30 seconds on any laptop (2020+), including 10 restarts for kernel optimization.

---

## 4. Integration with the Active Learning Loop

### Overview of the Three-Round Protocol

The active learning loop treats the 97-condition Sanchis-Calleja dataset as a foundation, then iteratively generates Engram-specific data to converge on optimal EC-generating conditions.

```
Round 0: Sanchis-Calleja 97 conditions (historical data)
   └── Fit GP to EC fraction + uncertainty
   └── GP recommends 24-condition CHIR x BMP4 dose matrix

Round 1: Run 24 conditions in lab → sequence → deconvolve cell types
   └── Update GP with new 24 data points (N_total = 121)
   └── GP uncertainty collapses in CHIR x BMP4 space
   └── Model identifies promising sub-region + untested morphogen combinations
   └── Recommend 12-16 refinement conditions

Round 2: Run refinement conditions → update model → converge
   └── N_total ≈ 133–137 conditions
   └── Posterior very tight around optimum
   └── Identify 2-3 final protocol candidates for validation
```

### Round 0: Train on Sanchis-Calleja 97 Conditions

**Data preparation from Sanchis-Calleja (Nature Methods 2025)**:
- Paper: "Generating human neural diversity with a multiplexed morphogen screen in organoids" (Cell Stem Cell, 2024) — 97 conditions × 100,538 single cells
- Morphogens screened: CHIR99021, XAV939, SHH, purmorphamine, FGF8, RA, BMP4, BMP7, cyclopamine
- Each cell was barcoded (10x hashing) to its condition → gives per-condition cell type composition
- From this: extract **cell type fraction** per condition (e.g., `EC_fraction = n_EC_cells / n_total_cells`)

**Key challenge**: The Sanchis-Calleja screen used neural organoids optimized for neural cell types. EC (endothelial cell) specification is not their primary readout, so EC fractions may be low or variable. This is actually useful: it defines the **baseline** we're trying to improve beyond.

```python
# Round 0: Fit GP on 97 historical conditions
# X_train: (97, D) — normalized morphogen conditions from the paper
# y_EC: (97,) — EC fraction per condition (from cell type deconvolution)

gpr_r0 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,
                                    normalize_y=True, random_state=42)
gpr_r0.fit(X_train_scaled, y_EC)

# Identify the 24-condition CHIR x BMP4 plate layout
# Strategy: 6 CHIR doses x 4 BMP4 doses = 24 conditions
# Choose doses to maximize combined EI + space filling
# Run full GP recommendation script (Section 3.3) to generate plate map
```

**What the Round 0 GP looks like**: With 97 scattered conditions across 8 morphogens, the GP will have:
- **High uncertainty** in most of the input space (most combinations untested)
- **Low uncertainty** near tested conditions
- **Non-trivial predictions** for pairwise combinations (additive kernel captures synergy)

The key output from Round 0 is not the predicted optimum — it is which region of the dose space to **densely sample** in Round 1.

### Round 1: CHIR x BMP4 24-Condition Dose Matrix

The GP (or manual inspection of Round 0 results) should motivate focusing on CHIR x BMP4 because:
- WNT signaling (CHIR) and BMP signaling (BMP4) are the primary determinants of mesoderm patterning, which is upstream of endothelial fate
- The Sanchis-Calleja screen likely sampled sparse points in this 2D subspace
- A dense 6x4 grid fills this subspace with predictable cost (24 organoid wells ≈ 1 plate)

**Recommended plate layout** (output of the BO recommendation system):

```
CHIR x BMP4 Dose Matrix (Round 1):
CHIR (µM): 0.0, 0.5, 1.0, 2.0, 3.0, 5.0
BMP4 (ng/mL): 0.0, 5.0, 10.0, 25.0

Total conditions: 24
Timing: CHIR days 0-3; BMP4 days 0-9 (based on Sanchis-Calleja timing windows)
Readout: scRNA-seq with cell hashing (multiplex all 24 conditions in 1 run)
Target output: EC fraction per condition (+ neuronal contamination)
```

After sequencing, update the GP:
```python
# Round 1 update: add new data
X_new = torch.tensor(round1_conditions, dtype=torch.float64)
y_new = torch.tensor(round1_EC_fractions, dtype=torch.float64).unsqueeze(-1)

X_combined = torch.cat([X_train, X_new], dim=0)  # (121, D)
Y_combined = torch.cat([Y_train, y_new], dim=0)    # (121, 1)

# Refit GP on combined data
gp_r1 = SingleTaskGP(X_combined, Y_combined)
mll_r1 = ExactMarginalLogLikelihood(gp_r1.likelihood, gp_r1)
fit_gpytorch_mll(mll_r1)
```

### Round 2: Refinement Based on Posterior Update

After Round 1, the CHIR x BMP4 subspace will have much lower uncertainty. The Round 2 recommendation system should:

1. **Identify the posterior optimum** in the CHIR x BMP4 space (the condition with highest predicted EC fraction + acceptable uncertainty)
2. **Expand to adjacent morphogens**: Does adding a small amount of VEGF, FGF2, or Notch signaling modulator improve EC fraction further?
3. **Probe timing effects**: Does the optimal CHIR x BMP4 combination change if applied at days 3-9 instead of days 0-9?

```python
# Round 2: Query optimum from Round 1 GP
best_condition_idx = Y_combined.argmax()
best_condition = X_combined[best_condition_idx]

# Add a 3rd dimension: timing window
# x_r2 = [CHIR, BMP4, timing_window]  where timing_window ∈ {0, 1, 2}
# Fix CHIR and BMP4 near Round 1 optimum; optimize timing and any add-on morphogen

# Example: optimize timing + VEGF concentration near the optimum
# Expand X to include VEGF concentration as 3rd dimension
```

### Uncertainty Collapse Pattern Across Rounds

The intuition for how posterior uncertainty evolves:

```
Round 0 (N=97):
   Uncertainty σ(x): HIGH everywhere in unexplored subspaces
   Effective posterior knowledge: covers 8 morphogens sparsely

Round 1 (N=97+24=121):
   Uncertainty σ(x): LOW in CHIR ∈ [0,5]µM x BMP4 ∈ [0,25] ng/mL
   Uncertainty σ(x): HIGH in all other subspaces (SHH, RA, etc.)
   GP now provides tight confidence intervals in the dense 2D subspace
   95% CI width ≈ 0.05 EC fraction (from ~0.15 in Round 0 to ~0.05 in Round 1)

Round 2 (N=121+16=137):
   Uncertainty σ(x): VERY LOW near optimum
   Model has converged on best CHIR x BMP4 conditions
   Adding timing/VEGF dimensions opens new uncertain dimensions to probe

Convergence criterion: when the 95% CI of the posterior maximum < 0.02 EC fraction,
stop and validate the top 2-3 conditions with triplicate experiments.
```

**Mathematical description of uncertainty collapse**: After adding `m` observations in a dense region, the posterior variance scales approximately as:
- σ²(x) ∝ σ²_prior × (1 - k(X_new, x)^T K(X_new, X_new)^{-1} k(X_new, x))
- As X_new fills the neighborhood of x, the Gram matrix term approaches 1, collapsing σ²(x) → 0

---

## 5. Comparison with Simpler Baselines

### 5.1 Linear Additive Model

The simplest baseline assumes morphogen effects are completely independent and additive:

```python
# Linear additive model: fit independent effect of each morphogen
from sklearn.linear_model import Ridge

def fit_linear_additive(X, y):
    """
    Fits: EC_fraction = β0 + β_CHIR*CHIR + β_BMP4*BMP4 + β_SHH*SHH + ...
    No interaction terms. Assumes independent morphogen effects.
    """
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    return model

linear_model = fit_linear_additive(X_train, y_EC)
y_linear_pred = linear_model.predict(X_novel_scaled)

# What the linear model cannot capture:
# - CHIR=3µM + BMP4=25 ng/mL may have super-additive effect
# - CHIR=5µM may saturate and have diminishing returns (nonlinear)
# - Low doses of multiple morphogens may act through distinct pathways
```

**When the linear additive model wins**:
- N_training >> N_parameters (you have lots of data)
- The underlying biology truly is additive (independent signaling pathways)
- You need an interpretable result: "each µM of CHIR adds X% EC fraction"
- You have >500 training conditions — the GP doesn't add much in this regime

**When the linear additive model fails** (GP wins):
- N < 200 training points and D > 3 dimensions (GP is better calibrated)
- Dose-response curves are sigmoidal or non-monotonic
- Combination effects exist (CHIR x BMP4 synergy)
- You need uncertainty estimates for novel conditions (linear model has no built-in UQ)

### 5.2 Cosine Similarity Lookup (Nearest Neighbor)

The simplest possible "model" for predicting a novel condition:

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def cosine_lookup(X_train, y_train, X_query, top_k=3):
    """
    For each query condition, find the k most similar training conditions
    by cosine similarity and return their weighted-average EC fraction.

    No model fitting. No uncertainty. Just lookup.
    """
    # Cosine similarity: measures angle between condition vectors
    # (assumes concentrations are all positive, which they are)
    similarities = cosine_similarity(X_query, X_train)  # (n_query, n_train)

    predictions = []
    for sim_row in similarities:
        top_k_idx = np.argsort(sim_row)[-top_k:]
        weights = sim_row[top_k_idx]
        weights = weights / weights.sum()  # normalize
        weighted_pred = (weights * y_train[top_k_idx]).sum()
        predictions.append(weighted_pred)

    return np.array(predictions)

y_cosine_pred = cosine_lookup(X_train_scaled, y_EC, X_novel_scaled, top_k=3)
```

**When cosine similarity wins**:
- You have a very dense training set covering the entire space
- The query point is very close to observed training points
- Speed is critical (lookup is O(N × D), much faster than GP inference)
- The training data already contains conditions very similar to what you want to test

**When cosine similarity fails** (GP wins):
- The query condition is genuinely novel (far from all training points)
- You need uncertainty quantification ("how confident are we?")
- The training set is sparse relative to the input space dimension
- You want to design the **next best experiment** (cosine lookup has no acquisition function)

### 5.3 When Does the GP Outperform? Empirical Evidence

From the literature directly relevant to this use case:

**GPerturb vs. deep learning (Xing & Yau 2025)**: GPerturb achieves competitive performance with state-of-the-art methods (GEARS, CPA, scGPT) without latent embeddings, using only sparse GP regression. In the sparse-data regime that defines organoid protocol optimization (<200 conditions), this is the relevant comparison.

**Deep learning vs. linear baselines (Ahlmann-Eltze et al., Nature Methods 2025)**: None of five foundation models (scGPT, scFoundation, GEARS, CPA, etc.) outperformed the simple additive baseline for predicting gene expression after perturbation. This does NOT mean GPs are worse — GPs are fundamentally different from these models. The key advantage of GPs is **calibrated uncertainty**, which deep learning models lack.

**BO vs. DoE (Narayanan et al., Nature Communications 2025)**: BO-based iterative experimental design identified optimal cell culture media conditions using 3-30× fewer experiments than standard Design of Experiments. With 24-experiment batches and 3 rounds (72 total), this maps directly to the 97-condition → 121 → 137 condition protocol.

**Rule of thumb for when GP beats linear/cosine baselines**:

| Condition | Linear Additive | Cosine | GP |
|-----------|-----------------|--------|-----|
| N < 50 training points | Poor | Poor | Best |
| N = 50-200 training points | Decent | Decent | Best |
| N > 500 training points | Often competitive | Often competitive | Marginal advantage |
| D > 5 dimensions | Struggles (need more data) | Struggles | Best with ARD |
| Non-linear dose-response | Fails | OK if dense | Best |
| Need uncertainty estimates | No | No | Best |
| Need next-experiment design | No | No | Best (via acquisition function) |
| Combination effects (synergy) | Fails | OK | Best with additive kernel |
| Training data has gaps | Extrapolates linearly | Fails | Best (principled extrapolation) |

**The decisive advantage of GPs in the organoid protocol design context**: With 97 conditions spread across 8 morphogens, the training data is very sparse in the 8-dimensional space. The GP's **kernel encodes the prior belief that nearby conditions have similar outputs** — allowing informed interpolation and extrapolation that neither the linear model nor cosine similarity can provide. The GP also gives calibrated uncertainty estimates that directly feed into the acquisition function to design the next experiment.

---

## 6. Summary: Implementation Decision Tree

```
Do you need next-experiment design (active learning)?
├── YES → Use GP + acquisition function (BoTorch or scikit-learn + manual EI)
│   ├── <10 min on laptop, single objective → scikit-learn GPR + manual EI grid search
│   ├── Multi-objective (EC + off-targets) → BoTorch qLogNEHVI + ModelListGP
│   └── Deep uncertainty characterization → GPerturb (slower, richer)
└── NO → Use simpler model for speed/interpretability
    ├── Need uncertainty per gene → GPerturb
    ├── Need fast lookup for known conditions → Cosine similarity
    └── Need interpretable coefficients → Linear additive model (Ridge)

Kernel choice:
├── Single morphogen → MaternKernel(nu=2.5)
├── 2-4 morphogens, possible synergy → AdditiveMaternGP(max_degree=2)
└── 5+ morphogens → SingleTaskGP with MaternKernel(ard_num_dims=D)

Acquisition function:
├── Round 0-1 (exploration) → UCB(beta=2.0) or LogEI
├── Round 2+ (exploitation) → LogExpectedImprovement
└── Multi-objective → qLogNoisyExpectedHypervolumeImprovement
```

---

## 7. References

1. Xing H, Yau C. "GPerturb: Gaussian process modelling of single-cell perturbation data." *Nature Communications* 16, 5423 (2025). https://doi.org/10.1038/s41467-025-61165-7
   - GitHub: https://github.com/hwxing3259/GPerturb

2. Sanchis-Calleja F, et al. "Generating human neural diversity with a multiplexed morphogen screen in organoids." *Cell Stem Cell* (2024). https://www.cell.com/cell-stem-cell/abstract/S1934-5909(24)00378-3
   - The 97-condition dataset underpinning Round 0 of the active learning loop.
   - Related method paper: "Systematic scRNA-seq screens profile neural organoid response to morphogens." *Nature Methods* (2025). https://www.nature.com/articles/s41592-025-02927-5

3. Narayanan H, et al. "Accelerating cell culture media development using Bayesian optimization-based iterative experimental design." *Nature Communications* 16, 6055 (2025). https://www.nature.com/articles/s41467-025-61113-5
   - Code: https://zenodo.org/records/15466161
   - Most directly analogous work: BO applied to biological media optimization, 3-30× more efficient than DoE.

4. Tosh C, et al. "A Bayesian active learning platform for scalable combination drug screens (BATCHIE)." *Nature Communications* 16, 156 (2025). https://www.nature.com/articles/s41467-024-55287-7
   - GitHub: https://github.com/tansey-lab/batchie
   - Architecture for sequential batch design — directly adaptable to morphogen screening.

5. Ahlmann-Eltze C, Huber W, Anders S. "Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines." *Nature Methods* 22, 1657–1661 (2025). https://www.nature.com/articles/s41592-025-02772-6
   - Critical context: benchmarks establishing that GP/additive models are competitive with deep learning for perturbation prediction.

6. BoTorch documentation: https://botorch.org/docs/
   - Multi-objective BO: https://botorch.org/docs/multi_objective/
   - Tutorial: qEHVI, qNEHVI: https://botorch.org/docs/tutorials/multi_objective_bo/

7. GPyTorch documentation — Additive/product kernels: https://docs.gpytorch.ai/en/v1.14/examples/00_Basic_Usage/kernels_with_additive_or_product_structure.html

8. scikit-learn GaussianProcessRegressor: https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html

9. Duvenaud D, et al. "Additive Gaussian Processes." *NIPS* (2011). — Foundation for additive kernel design capturing morphogen combination effects.

10. Bayesian active learning strategy for sequential experimental design in systems biology. *BMC Systems Biology* (2014). https://pmc.ncbi.nlm.nih.gov/articles/PMC4181721/
    - Theoretical foundation for the round-by-round uncertainty collapse.

---

## 8. Comprehensive Literature Review: GP-BO for Biological Protocol Optimization

> **Date**: 2026-03-07
> **Purpose**: Answers 10 research questions on the feasibility and landscape of GP-BO for brain organoid differentiation protocol optimization, starting from 97 published conditions (Sanchis-Calleja 2025).

---

### 8.1 Has Anyone Used GP-BO for Cell Differentiation Protocol Optimization?

**Yes, multiple groups, with strong results.** The most directly relevant examples:

**Narayanan et al. (2025), Nature Communications** -- "Accelerating cell culture media development using Bayesian optimization-based iterative experimental design"
- Used GP-based BO with iterative experimental design for two cell culture applications: (1) optimizing cytokine supplementation to maintain viability and phenotypic distribution of human PBMCs ex vivo, and (2) optimizing recombinant protein production in *K. phaffii*.
- Achieved optimized conditions using **3--30x fewer experiments** than DoE approaches (OFAT, Box-Behnken, CCD, full factorial, fractional factorial).
- For the PBMC application: optimized a blend of 4 commercial media (DMEM, AR5, XVIVO, RPMI) to maximize viability, then optimized 8 cytokines (IL-2, IL-3, IL-4, IL-7, IL-12, IL-15, IL-21, BAFF) to maintain lymphocyte subpopulation balance.
- For PBMC viability: 24 total experiments (4 iterations of 6), improved viability from ~60% to 75--80%.
- For cytokines (8 design factors): 12 total experiments identified a condition maintaining T, B, and NK cell balance. A DoE equivalent would require 80+ experiments (BBD) to 6,561 experiments (full factorial 3^8).
- For K. phaffii protein production: optimized 9 factors (carbon sources including categorical variables) with 3 rounds. Achieved 10--30x reduction vs. DoE.
- **GP kernel used**: Standard Gaussian Process with Matern kernel. Handled continuous, discrete, AND categorical variables (carbon source type). Used BoTorch/GPyTorch implementation.
- **Acquisition function**: Expected Improvement with exploration-exploitation trade-off.
- **Directly analogous to Engram's use case**: morphogen concentrations are continuous design factors; cell type composition is the readout.
- Code: https://zenodo.org/records/15466161

**Claes et al. (2024), Biotechnology and Bioengineering** -- "Bayesian cell therapy process optimization"
- Evaluated noisy, parallel BO for cell therapy manufacturing.
- Tested on in silico bioprocesses with increasing noise levels and parallel batch sizes.
- Found BO outperforms DoE even with significant experimental variability (biological noise).
- In vitro validation on mesenchymal stromal cell expansion: BO with parallel batches of 6 experiments per round converged in 3--4 rounds.
- Used noisy GP with Matern 5/2 kernel and qExpectedImprovement for batch recommendations.

**Cosenza et al. (2023), Engineering in Life Sciences** -- "Multi-objective Bayesian algorithm automatically discovers low-cost high-growth serum-free media"
- Applied multi-objective BO to optimize serum-free media for cellular agriculture (bovine satellite cells).
- 14 media components (14-dimensional input space).
- Used multi-information source BO: cheap rapid growth assays + expensive but accurate cell counts.
- Identified Pareto-optimal media formulations balancing growth rate and cost.
- Demonstrated that BO with GP surrogates is feasible even in 14D with limited experiments.

**Sambu (2014), PeerJ Preprints** -- "A Bayesian approach to optimizing stem cell cryopreservation protocols"
- Early example of BO for stem cell protocol optimization.
- Optimized cryopreservation parameters (cooling rate, cryoprotectant concentration) using GP surrogate.

---

### 8.2 GPerturb (Xing & Yau 2025, Nature Communications): How It Works and Applicability

**Full reference**: Xing H, Yau C. "GPerturb: Gaussian process modelling of single-cell perturbation data." *Nature Communications* 16, 5423 (2025).

**What it does**: GPerturb is a hierarchical Bayesian model using Gaussian process regression to estimate gene-level perturbation effects from single-cell CRISPR screening data (Perturb-seq, CROP-seq).

**Core architecture**:
- Decomposes observed gene expression as: `y_i = f_basal(cell_info_i) + f_perturbation(perturbation_i) + noise`
- `f_basal` and `f_perturbation` are modeled by separate GPs
- Perturbation effects are **sparse**: a binary indicator variable per gene determines whether a perturbation affects that gene at all
- Provides uncertainty estimates for both the **presence** and **magnitude** of perturbation effects

**Two observation models**:
- GPerturb-Gaussian: for log-transformed continuous expression
- GPerturb-ZIP: for raw UMI counts (zero-inflated Poisson)

**Benchmark performance** (from paper):
- Compared against CPA, GEARS, SAMS-VAE on Replogle et al. genome-wide Perturb-seq dataset
- GPerturb achieves **competitive prediction performance** with state-of-the-art deep learning methods
- Key advantage: interpretability. The sparse binary indicators directly identify which genes are affected by each perturbation.
- Handles both discrete (gene knockout) and continuous (drug dose) perturbations

**Applicability to Engram**:
- **Partially applicable**. GPerturb excels at characterizing *what happened* in a screen (retrospective analysis) -- identifying which genes respond to which morphogens.
- It does NOT natively implement acquisition functions or recommend next experiments.
- To use for active learning: run GPerturb on Sanchis-Calleja data to identify gene-level morphogen responses, then feed the uncertainty estimates into a separate BO layer (BoTorch) for experiment recommendation.
- The additive GP structure (basal + perturbation) maps naturally to the organoid setting where cell type identity is the basal component and morphogen exposure is the perturbation.

**Limitation for Engram**: GPerturb operates at the gene expression level, not the cell type composition level. Engram's objective is to optimize cell type fractions (e.g., maximize EC percentage), which is a derived summary statistic from the single-cell data. A simpler GP directly on cell type fractions may be more practical for the BO loop, while GPerturb provides deeper mechanistic insight.

---

### 8.3 BATCHIE (Tosh et al. 2025, Nature Communications): Active Learning with 4% of Search Space

**Full reference**: Tosh C, Tec M, White JB, et al. "A Bayesian active learning platform for scalable combination drug screens." *Nature Communications* 16, 156 (2025).

**The problem**: Combination drug screens are intractable because the number of experiments grows as n x m^d x t^d (n conditions, m drugs, t doses, d-way combinations). A pairwise screen of 100 drugs at 5 doses over 50 cell lines = 6.2M experiments.

**How BATCHIE achieves 4%**:

1. **Bayesian tensor factorization model**: The core model embeds each cell line, drug-dose, and drug interaction into a low-dimensional space (dimension d). For drug-doses i and j on cell line k, the logit-viability is:
   ```
   mu_ijk = sum_t (v1_t^(i) + v1_t^(j) + v2_t^(i) * v2_t^(j)) * u_t^(k)
   ```
   where v1 captures individual effects, v2 captures interaction effects, and u captures cell line sensitivity.

2. **PDBAL (Probabilistic Diameter-based Active Learning)**: Instead of random or fixed-design experiments, BATCHIE selects each batch to **minimize the expected distance between any two posterior samples** after observing the batch outcomes. This is a theoretically near-optimal criterion with proven guarantees.

3. **Sequential batch design**: Start with a seed batch covering every drug and cell line at least once. Then iteratively: fit model, draw posterior samples, score candidate experiments by expected information gain, select the most informative batch, run experiments, update model.

4. **Submodular batch selection**: Individual experiment scores are aggregated into batch scores using a submodular approach, ensuring the batch is maximally informative as a whole (not just individually informative experiments).

**Prospective validation results**:
- Library: 206 drugs, 16 cancer cell lines (pediatric sarcomas)
- Total possible pairwise experiments: ~1.4M
- BATCHIE explored only **4% of the space** (~56K experiments)
- Model accurately predicted unseen combinations
- Identified top combinations for Ewing sarcoma, all 10 validated hits confirmed effective
- Top hit: PARP inhibitor (talazoparib) + topoisomerase I inhibitor (topotecan) -- already in Phase II clinical trials

**Relevance to Engram**:
- BATCHIE's batch design philosophy is directly transferable: instead of fixed factorial designs for morphogen combinations, design each batch adaptively based on previous results.
- The tensor factorization model could be adapted: replace drug embeddings with morphogen embeddings, cell line embeddings with organoid batch/cell line embeddings.
- BATCHIE is open source: https://github.com/tansey-lab/batchie
- **Key difference from standard GP-BO**: BATCHIE uses active learning (model the whole space) rather than Bayesian optimization (find a single optimum). For Engram, where the goal is to find the best protocol (optimization), standard BO with EI/UCB is more appropriate. BATCHIE would be more useful if Engram wanted to characterize the full morphogen-response landscape.

---

### 8.4 Narayanan et al. 2025: BO 3--30x Fewer Experiments Than DoE

**Detailed results breakdown**:

**Application 1 -- PBMC media blend optimization (4 continuous factors, constrained)**:
- Design space: mixture of 4 commercial media (DMEM, AR5, XVIVO, RPMI), constrained to sum to 100%
- Objective: maximize PBMC viability at 72h
- BO approach: 4 iterations x 6 experiments = 24 total experiments
- Result: viability improved from ~60% (individual media) to 75--80% (optimized blend)
- DoE comparison: OFAT would require 12 experiments (but no interactions), Box-Behnken Design 27, Central Composite Design 25
- **Reduction factor: ~1.5x vs. OFAT, ~1.1x vs. BBD** (modest for 4 factors)

**Application 2 -- PBMC cytokine optimization (8 factors with constraints)**:
- Design space: 8 cytokines (IL-2, IL-3, IL-4, IL-7, IL-12, IL-15, IL-21, BAFF) at multiple concentration levels
- Objective: maintain balanced lymphocyte subpopulation distribution
- BO approach: 12 total experiments (2 rounds x 6)
- Result: identified cytokine cocktail maintaining T, B, and NK cell homeostasis
- DoE comparison: BBD requires 81 experiments, CCD 81, Full Factorial (3^8) = 6,561, Fractional Factorial ~128
- **Reduction factor: ~7x vs. BBD, ~10x vs. Fractional Factorial, ~547x vs. Full Factorial**

**Application 3 -- K. phaffii carbon source optimization (9 factors including categorical)**:
- 9 design factors including continuous (concentration) and categorical (carbon source type: glucose, glycerol, methanol, sorbitol)
- Objective: maximize specific productivity of recombinant proteins
- 3 rounds of BO with 6 experiments per round = 18 total experiments
- DoE comparison: with categorical variables, DoE designs scale dramatically
- **Reduction factor: 10--30x vs. DoE** depending on the DoE variant

**Key technical details**:
- GP kernel: custom kernel handling mixed continuous/categorical variables
- Acquisition function: Expected Improvement
- Batch size: 6 experiments per round (practical limit of parallel bioreactors)
- Transfer learning: demonstrated that learning from one protein (RBDJ) could accelerate optimization of another (HSA, trastuzumab)
- **The 3--30x claim scales with dimensionality**: at 4D the advantage is modest (~3x), at 8--9D the advantage is dramatic (~30x). For Engram's 8-morphogen space, expect ~10--30x reduction.

---

### 8.5 Is 97 Datapoints Enough for a GP with ~8 Morphogen Dimensions?

**The Loeppky 10d Rule**: The standard rule of thumb from Loeppky, Sacks & Welch (2009, *Technometrics*) is that an effective initial experiment for GP regression requires approximately **n = 10 x d** observations, where d is the input dimension. For d=8, this gives **n = 80**.

**97 > 80, so yes, 97 datapoints is sufficient for an initial GP fit in 8 dimensions by this rule.**

However, several caveats apply:

**1. The 10d rule assumes space-filling designs**. The Sanchis-Calleja 97 conditions are NOT a space-filling design (Latin hypercube, Sobol sequence). They are hand-picked conditions probing specific hypotheses about neural regionalization. Coverage of the 8D morphogen space is likely uneven -- dense in some regions, sparse in others. The effective dimensionality may be lower (many conditions share the same morphogen values), which helps.

**2. Harari et al. (2018, *Statistica Sinica*)** revisited the 10d rule and showed it depends on the roughness of the response surface. For smooth functions (long lengthscales), fewer points suffice. For rough functions (short lengthscales), more are needed. Biological dose-response curves are typically smooth (sigmoidal), favoring fewer required points.

**3. Xu et al. (2024, arXiv:2402.02746)** -- "Standard Gaussian Process is All You Need for High-Dimensional Bayesian Optimization" -- systematically tested standard GP-BO up to 100 dimensions and found it performs surprisingly well even without dimensionality reduction, contradicting the widespread belief that GPs fail in high dimensions. The key insight: with Matern kernels and ARD lengthscales, the GP automatically identifies relevant dimensions and effectively operates in a lower-dimensional subspace.

**4. Binois & Wycoff (2022, HAL survey)** -- "A survey on high-dimensional Gaussian process modeling with application to Bayesian optimization" -- review structural assumptions for high-D GPs: variable selection, additive decomposition, linear embeddings. For d=8, no special high-dimensional techniques are needed. Standard GP with ARD is appropriate.

**5. Practical evidence from Narayanan et al. (2025)**: They started with batches as small as 6 experiments in 8--9 dimensional spaces and successfully optimized within 12--18 total experiments. The BO loop does not require the initial GP to be highly accurate -- it only needs to be good enough to identify the most informative next experiment.

**Bottom line on sample size**: 97 datapoints in 8 dimensions is above the 10d=80 threshold, and the iterative BO framework means the initial GP only needs to be "good enough" to guide the first round of experiments. After Round 1 (N=121) and Round 2 (N=137+), the GP will be well-conditioned in the subspace of interest. **97 is a comfortable starting point.**

---

### 8.6 Kernel Selection: What Do Practitioners Recommend?

**Consensus from the literature**:

| Kernel | When to Use | Evidence |
|--------|------------|----------|
| **Matern 5/2 + ARD** | Default for biological optimization. Assumes twice-differentiable responses. | Narayanan 2025, Claes 2024, Siska 2025 (bioprocess guide) all use Matern. BoTorch default. |
| **Matern 3/2** | When threshold/switch-like effects are expected (e.g., binary fate decisions at a morphogen threshold). Less smooth than 5/2. | Appropriate if CHIR has an all-or-nothing effect on WNT activation at a critical concentration. |
| **RBF (squared exponential)** | Only when you are confident the response is infinitely smooth. Rarely appropriate for biology. | Over-smooths; misses threshold effects. Avoid unless benchmarking. |
| **Additive Matern** | When morphogens act through mostly independent pathways. Decomposes into univariate + pairwise terms. | Duvenaud et al. 2011 (NIPS). Good for 5+ morphogens to reduce effective dimensionality. |
| **ARD (Automatic Relevance Determination)** | Always use with >3 input dimensions. Learns one lengthscale per morphogen; irrelevant morphogens get infinite lengthscales. | Standard in BoTorch SingleTaskGP. Critical for 8-morphogen screen to identify which morphogens matter. |

**Recommendation for Engram**:

- **Start with**: `SingleTaskGP` in BoTorch (uses `ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=8)) + WhiteKernel` by default)
- **After Round 1**: Inspect ARD lengthscales. If 3--4 morphogens have much shorter lengthscales than the rest, consider switching to an additive kernel over just those morphogens for better data efficiency.
- **Never use RBF** for dose-response data without benchmarking against Matern first.

**From the Siska et al. (2025) bioprocess engineering guide** (arXiv:2508.10642): "The Matern-5/2 kernel is the default choice for bioprocess optimization. It provides a good balance between flexibility and smoothness... ARD lengthscales are essential when the response surface has different sensitivity to different input dimensions, which is almost always the case in biological systems."

---

### 8.7 GP Approach vs. Foundation Models: Comparison for Small-Data Perturbation Prediction

**Key benchmark**: Ahlmann-Eltze, Huber & Anders (2025, *Nature Methods*) -- "Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines." Found that none of five foundation models (scGPT, GEARS, CPA, scFoundation, etc.) consistently outperformed a simple additive linear baseline for predicting gene expression after perturbation.

**Comparison table**:

| Method | Type | Training Data Needed | Handles <200 Conditions | Uncertainty Quantification | Interpretability | Active Learning Ready | Best For |
|--------|------|---------------------|------------------------|---------------------------|------------------|----------------------|----------|
| **GP (Matern + ARD)** | Probabilistic surrogate | 10d--20d points | Yes (designed for this) | Native, calibrated | High (lengthscales, ARD) | Yes (EI, UCB, qNEHVI) | Small-data optimization with experiment design |
| **GPerturb** | Hierarchical GP | ~100+ conditions | Yes | Native, gene-level | Very high (sparse indicators) | Partial (needs BO wrapper) | Gene-level perturbation characterization |
| **CellFlow** | Flow matching (generative) | 1000s of perturbations | No -- needs large training set | No native UQ | Low (neural network) | No | Large-scale in silico perturbation prediction |
| **scGPT** | Transformer foundation model | Pre-trained on 33M cells | Can fine-tune on small data | No native UQ | Low | No | Transfer learning across cell types |
| **Geneformer** | Transformer foundation model | Pre-trained on 30M cells | Can fine-tune on small data | No native UQ | Moderate (attention weights) | No | Predicting effects of gene perturbations |
| **GEARS** | GNN + gene knowledge graph | 100+ perturbations | Borderline | No native UQ | Moderate (gene graph) | No | Combinatorial gene perturbation prediction |
| **CPA** | Variational autoencoder | 100+ perturbations | Borderline | No native UQ | Moderate (disentangled latent) | No | Drug/perturbation dose-response |
| **Linear additive baseline** | Linear regression | 10d points | Yes | Via bootstrap/Bayesian LR | Very high | Partial (no acquisition function) | Benchmarking; interpretable coefficients |

**CellFlow details** (Klein et al., 2025, bioRxiv): CellFlow uses conditional flow matching to model single-cell phenotypes. It generates synthetic single-cell distributions for unseen perturbations. Benchmarked on cytokine stimulation, drug treatment, and genetic perturbation data. Achieves strong performance on large datasets but **requires thousands of perturbation conditions** for training -- far more than Engram's 97.

**scDFM** (Yu et al., 2026, ICLR): Distributional flow matching variant that models population-level shifts rather than cell-level correspondences. Same data hunger issue.

**Why GP wins for Engram's use case**:
1. **97 datapoints is tiny** -- deep learning and foundation models are data-hungry by design
2. **Calibrated uncertainty** is essential for experiment design -- GP provides it natively; neural models don't
3. **Interpretability** -- ARD lengthscales directly tell you which morphogens matter; neural network latent spaces don't
4. **Foundation models aren't trained on organoid morphogen data** -- scGPT/Geneformer are pre-trained on gene perturbation (CRISPR knockout) data, not morphogen dose-response. The transfer gap is large.
5. **Active learning requires uncertainty** -- without reliable uncertainty estimates, you cannot compute acquisition functions. This eliminates all deterministic neural models from the BO loop.

---

### 8.8 Multi-Objective BO with BoTorch: Has Anyone Used It for Cell Biology?

**Yes, several published examples:**

**Cosenza et al. (2023)**: Used multi-objective BO for serum-free media optimization in cellular agriculture. Optimized 14 media components simultaneously for growth rate and cost. Used BoTorch's multi-objective acquisition functions to identify Pareto-optimal formulations.

**Selega & Campbell (2023, TMLR)**: "Multi-objective Bayesian Optimization with Heuristic Objectives for Biomedical and Molecular Data Analysis Workflows." Applied MOBO to single-cell data analysis pipeline optimization, jointly optimizing clustering quality and computational cost. Demonstrated that MOBO with BoTorch's qNParEGO successfully navigates trade-offs in biomedical data analysis.

**BoTier (2025, arXiv:2501.15554)**: Introduced tiered composite objectives for scientific optimization, reflecting the hierarchy of preferences in real experiments (e.g., "first maximize yield, then minimize cost among equally good yields"). Applicable to organoid optimization where the primary objective is cell type composition and secondary objectives include viability, batch-to-batch consistency, and cost.

**BoTorch multi-objective capabilities** (fully supported):
- `qLogNoisyExpectedHypervolumeImprovement` (qLogNEHVI): the recommended acquisition function for noisy multi-objective optimization. Computes expected improvement in the hypervolume of the Pareto front.
- `qNParEGO`: random scalarization approach, simpler but less sample-efficient.
- `qLogEHVI`: for noiseless observations.
- Supports >3 objectives, though hypervolume computation becomes expensive above 5 objectives.

**Recommendation for Engram**: Use BoTorch's qLogNEHVI with 2--3 objectives:
1. Maximize target cell type fraction (e.g., EC or cortical neuron)
2. Minimize off-target fraction (e.g., undifferentiated cells)
3. (Optional) Maximize batch-to-batch consistency (minimize variance across replicates)

---

### 8.9 Known Failure Modes of GP-BO in Biology

**1. Non-stationarity**: GPs with standard stationary kernels (Matern, RBF) assume the response surface has the same smoothness everywhere. In biology, this can fail: the dose-response may be smooth at low concentrations but exhibit sharp transitions at critical thresholds (e.g., WNT activation above a CHIR threshold). **Mitigation**: Use Matern 3/2 instead of 5/2 if threshold effects are expected, or use input-dependent lengthscale models (deep kernel learning).

**2. Batch effects**: Organoid experiments across different batches (different passages, different operators, different media lots) introduce systematic variation that the GP does not model. If Round 1 data has a batch offset from the Sanchis-Calleja data, the GP will be miscalibrated. **Mitigation**: Include batch as a categorical covariate in the GP (either as a fixed effect or a separate kernel component). Narayanan et al. (2025) explicitly addressed this via their transfer learning framework.

**3. Cell line variability**: Different iPSC lines respond differently to morphogens. The Sanchis-Calleja screen used HES3 (one hESC line) and demonstrated variability when testing on other lines. A GP trained on HES3 data may not transfer to Engram's iPSC lines. **Mitigation**: Run a small pilot (6--12 conditions) on Engram's cell line, then use multi-task GP or transfer learning to incorporate the historical data with a cell-line-specific offset.

**4. Observation noise and measurement uncertainty**: scRNA-seq cell type deconvolution has inherent noise. If condition A has 500 cells and condition B has 50 cells, the EC fraction estimate for B is much noisier. Standard GP assumes homoscedastic noise. **Mitigation**: Use heteroscedastic GP (BoTorch's `HeteroskedasticSingleTaskGP`) or weight observations by cell count.

**5. Boundary effects and extrapolation**: GPs revert to the prior mean (and high uncertainty) far from training data. If the optimum is at the edge of the tested concentration range, the GP may not recommend going beyond it. **Mitigation**: Include a few "extreme" conditions in Round 1 at the boundaries of biologically plausible ranges.

**6. Curse of dimensionality (mitigated at d=8)**: With 97 points in 8D, the data is sparse. However, the ARD kernel handles this well by learning that only 3--4 morphogens are truly relevant, effectively reducing the problem to 3--4D. The Xu et al. (2024) paper showed standard GPs work well up to 100D, so d=8 is not problematic.

**7. Multi-modal response surfaces**: If the cell type composition landscape has multiple disjoint optima (e.g., dorsal cortex at one morphogen ratio, ventral forebrain at another), EI can get stuck exploiting one mode. **Mitigation**: Use Thompson sampling or UCB with high beta in early rounds to explore broadly.

**8. Temporal dynamics**: Morphogen timing matters (days 0--3 vs. 3--9 vs. 9--21). The GP treats timing as just another input dimension, but the biological relationship between timing and dose is hierarchical (timing determines competence windows). **Mitigation**: Consider a structured kernel that separates timing from concentration effects.

---

### 8.10 End-to-End GP-BO to Wet Lab Validation to GP Update Cycle: Published Examples

**SAMPLE Platform (Rapp, Bremer & Romero, 2024, Nature Chemical Engineering)**:
- **The gold standard for closed-loop GP-BO in biology.**
- Self-driving Autonomous Machines for Protein Landscape Exploration (SAMPLE)
- Deployed 4 autonomous agents to engineer glycoside hydrolase enzymes for enhanced thermostability
- Each agent: GP learns sequence-function relationship -> designs new proteins -> automated robotic system synthesizes and tests them -> results fed back to GP -> repeat
- All 4 agents converged on enzymes >12C more thermostable than starting sequences
- Ran autonomously for multiple rounds (build-test-learn cycles)
- Used GP with Matern kernel for surrogate model, UCB acquisition function
- **Key insight**: Despite individual differences in search behavior (some agents explored more, others exploited), all converged successfully. This demonstrates the **robustness** of GP-BO to different exploration strategies.
- Published in the inaugural issue of Nature Chemical Engineering; 155 citations, 55k accesses.

**Coutant et al. (2019), PNAS** -- "Closed-loop cycles of experiment design, execution, and learning accelerate systems biology model development in yeast"
- Not GP-BO per se, but a closed-loop experiment design framework for yeast systems biology
- Demonstrated that iterative model-guided experimentation converges faster than batch experimentation
- 3--5 rounds of design-execute-learn reduced model uncertainty by >50% vs. equivalent number of random experiments

**Helleckes et al. (2025, arXiv:2508.10970)** -- "Holistic Bioprocess Development Across Scales Using Multi-Fidelity Batch Bayesian Optimization"
- Multi-fidelity BO for bioprocess development: uses cheap small-scale experiments (microtiter plates) to inform expensive large-scale experiments (bioreactors)
- Gaussian process with multi-fidelity kernel
- Demonstrated 2--5x reduction in total experimental cost by intelligently allocating experiments across scales
- Directly relevant if Engram uses 96-well plates for screening and transitions to larger organoid cultures for validation

**Liu et al. (2025, npj Computational Materials)** -- "Active oversight and quality control in standard Bayesian optimization for autonomous experiments"
- Addresses the "human-in-the-loop" aspect: how to maintain quality control when BO recommends experiments autonomously
- Proposes checkpoints where the experimentalist reviews BO recommendations before execution
- Relevant for Engram: the wet lab team should review GP recommendations before committing to expensive organoid experiments

---

### 8.11 Comparison Summary Table

| Paper | Year | Method | Domain | Input Dims | Starting N | Total Experiments | Key Result | Code Available |
|-------|------|--------|--------|-----------|-----------|-------------------|------------|----------------|
| Narayanan et al. | 2025 | GP-BO (BoTorch) | Cell culture media | 4--9 | 6 | 12--24 | 3--30x fewer than DoE | Yes (Zenodo) |
| Tosh et al. (BATCHIE) | 2025 | Bayesian active learning | Drug combinations | 206 drugs x 16 lines | Seed batch | 4% of 1.4M space | Accurate prediction + validated hits | Yes (GitHub) |
| Xing & Yau (GPerturb) | 2025 | Hierarchical GP | scRNA-seq perturbation | Gene-level | 100+ perturbations | Retrospective | Competitive with DL, interpretable | Yes (GitHub) |
| Rapp et al. (SAMPLE) | 2024 | GP-BO, autonomous | Protein engineering | Sequence space | ~100 variants | 4 autonomous rounds | +12C thermostability | Yes |
| Claes et al. | 2024 | Noisy parallel BO | Cell therapy | 4--6 | 6 | 18--24 | BO > DoE with noise | No |
| Cosenza et al. | 2023 | Multi-objective BO | Serum-free media | 14 | ~20 | ~60 | Pareto-optimal formulations | No |
| Sanchis-Calleja et al. | 2025 | Systematic screen | Neural organoid morphogens | 8+ morphogens | -- | 97 conditions | Morphogen competence windows mapped | Data available |

---

### 8.12 Bottom Line: Is GP-BO the Right Approach for 97 Datapoints with 8 Morphogen Dimensions?

**Yes. GP-BO is the right approach, and this is close to the ideal use case for it.**

**Arguments for GP-BO**:

1. **Sample size is sufficient**: 97 > 10d = 80 (Loeppky rule). The initial GP will be well-enough calibrated to guide the first round of adaptive experiments.

2. **The literature validates exactly this setting**: Narayanan et al. (2025) demonstrated GP-BO with 8--9 design factors starting from 6 initial experiments and achieving convergence in 12--24 total. Engram starts with 97 -- far more than typical BO starting points.

3. **Foundation models are NOT competitive here**: With only 97 conditions, deep learning models (CellFlow, scGPT, GEARS) lack sufficient training data. The Ahlmann-Eltze (2025) benchmark showed they don't even beat linear baselines on perturbation prediction.

4. **Calibrated uncertainty is essential**: The entire value proposition of active learning depends on reliable uncertainty estimates. GPs provide these natively; neural models don't.

5. **ARD handles the 8D input space well**: The GP will automatically learn which morphogens matter (short lengthscale) and which don't (long lengthscale), effectively reducing the problem to the 3--4 most relevant dimensions.

6. **The iterative framework is proven**: Multiple publications (Narayanan, SAMPLE, Claes) have demonstrated the GP -> wet lab -> GP update cycle successfully closes the loop.

**Risk factors to manage**:

1. **Batch effects between Sanchis-Calleja data and Engram's lab**: Different cell lines, different operators, different media lots. Mitigate with a small pilot experiment (6--12 conditions) on Engram's cell line before trusting the GP's predictions.

2. **Non-stationarity**: Morphogen response may have sharp thresholds the GP smooths over. Start with Matern 5/2, switch to Matern 3/2 if residuals suggest threshold effects.

3. **Output metric**: The choice of what to optimize matters. Cell type fraction from scRNA-seq is noisy. Consider using multiple deconvolution methods and averaging, or use surface marker FACS as a cheaper intermediate readout for the BO loop.

**Recommended implementation path**:

```
Round 0: Fit GP (Matern 5/2, ARD) on Sanchis-Calleja 97 conditions
         -> Inspect ARD lengthscales to identify key morphogens
         -> Generate 24-condition recommendation for Round 1

Round 1: Run 24 conditions on Engram cell line
         -> Update GP (N=121)
         -> Uncertainty collapses in key morphogen subspace
         -> Generate 12-16 refinement conditions

Round 2: Run refinement conditions (N=133-137)
         -> GP posterior tight around optimum
         -> Validate top 2-3 conditions with triplicate experiments

Total: ~60 new experiments to identify optimized protocol
       (vs. 500-6000+ for DoE approaches in 8D)
```

**Software stack**: BoTorch + GPyTorch for the GP and acquisition functions. scikit-learn for rapid prototyping. GPerturb for deeper gene-level mechanistic analysis once the optimal conditions are identified.

---

### 8.13 Additional References

11. Rapp JT, Bremer BJ, Romero PA. "Self-driving laboratories to autonomously navigate the protein fitness landscape." *Nature Chemical Engineering* 1, 97--107 (2024). https://doi.org/10.1038/s44286-023-00002-4

12. Xu Z, Zhe S. "Standard Gaussian Process is All You Need for High-Dimensional Bayesian Optimization." *JMLR* (2024). https://arxiv.org/abs/2402.02746

13. Loeppky JL, Sacks J, Welch WJ. "Choosing the Sample Size of a Computer Experiment: A Practical Guide." *Technometrics* 51(4), 366--376 (2009).

14. Binois M, Wycoff N. "A survey on high-dimensional Gaussian process modeling with application to Bayesian optimization." HAL (2022). https://inria.hal.science/hal-03419959

15. Ahlmann-Eltze C, Huber W, Anders S. "Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines." *Nature Methods* 22, 1657--1661 (2025).

16. Klein D, Fleck JS, et al. "CellFlow enables generative single-cell phenotype modeling with flow matching." *bioRxiv* (2025). https://doi.org/10.1101/2025.04.11.648220

17. Li C, et al. "Benchmarking AI Models for In Silico Gene Perturbation of Cells." *bioRxiv* (2024). https://doi.org/10.1101/2024.12.20.629581

18. Cosenza Z, Block DE, Baar K, et al. "Multi-objective Bayesian algorithm automatically discovers low-cost high-growth serum-free media for cellular agriculture application." *Engineering in Life Sciences* 23(8) (2023).

19. Siska M, Pajak E, et al. "A Guide to Bayesian Optimization in Bioprocess Engineering." arXiv:2508.10642 (2025).

20. Harari O, Bingham D, Dean A, Higdon D. "Computer experiments: prediction accuracy, sample size and model complexity revisited." *Statistica Sinica* 28, 899--919 (2018).

21. Sanchis-Calleja F, et al. "Systematic scRNA-seq screens profile neural organoid response to morphogens." *Nature Methods* 23(2), 465--478 (2025). https://doi.org/10.1038/s41592-025-02927-5

22. Claes E, et al. "Bayesian cell therapy process optimization." *Biotechnology and Bioengineering* (2024). https://pubmed.ncbi.nlm.nih.gov/38372656/

23. Selega A, Campbell KR. "Multi-objective Bayesian Optimization with Heuristic Objectives for Biomedical and Molecular Data Analysis Workflows." *TMLR* (2023).

24. Martens A, et al. "Holistic Bioprocess Development Across Scales Using Multi-Fidelity Batch Bayesian Optimization." arXiv:2508.10970 (2025).
