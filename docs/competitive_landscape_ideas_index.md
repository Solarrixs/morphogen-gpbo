# Competitive Landscape: Implementable Ideas Index

Cross-references actionable ideas extracted from 9 papers, ranked by priority for the morphogen-gpbo pipeline.

## Source Papers

| Paper | File | Ideas | DOI |
|---|---|---|---|
| DeMeo et al. 2025 (Science) | [ideas_from_demeo_2025.md](ideas_from_demeo_2025.md) | 6 | 10.1126/science.adi8577 |
| Narayanan et al. 2025 (Nat Comm) | [ideas_from_narayanan_2025.md](ideas_from_narayanan_2025.md) | 6 | 10.1038/s41467-025-61113-5 |
| Cosenza et al. 2022 (Biotech Bioeng) | [ideas_from_cosenza_2022.md](ideas_from_cosenza_2022.md) | 6 | 10.1002/bit.28132 |
| McDonald et al. 2025 (ACS Cent Sci) | [ideas_from_mcdonald_2025.md](ideas_from_mcdonald_2025.md) | 6 | 10.1021/acscentsci.4c01991 |
| BATCHIE / Tosh et al. 2025 (Nat Comm) | [ideas_from_batchie_2025.md](ideas_from_batchie_2025.md) | 5 | 10.1038/s41467-024-55287-7 |
| GPerturb / Xing & Yau 2025 (Nat Comm) | [ideas_from_gperturb_2025.md](ideas_from_gperturb_2025.md) | 5 | 10.1038/s41467-025-61165-7 |
| NAIAD / Qin et al. 2025 (ICML) | [ideas_from_naiad_2025.md](ideas_from_naiad_2025.md) | 5 | arXiv:2411.12010 |
| CellFlow / Klein et al. 2025 (bioRxiv) | [ideas_from_cellflow_2025.md](ideas_from_cellflow_2025.md) | 7 | 10.1101/2025.04.11.648220 |
| Sanchis-Calleja et al. 2025 (Nat Methods) | [ideas_from_sanchis_calleja_2025.md](ideas_from_sanchis_calleja_2025.md) | 6 | 10.1038/s41592-025-02927-5 |

**Total: 52 ideas across 9 papers**

---

## Top Priority Ideas (CRITICAL / HIGH)

### Must-Do Before Round 2

| # | Idea | Source Paper | Priority | Why Critical |
|---|---|---|---|---|
| 1 | **Cross-fidelity correlation validation gate** | McDonald 2025 | CRITICAL | MF-BO fails if fidelity correlation is too weak or too strong. We have ZERO validation. |
| 2 | **Explicit cost ratios in acquisition** | McDonald 2025 | HIGH | Fidelity labels (0.0/0.5/1.0) carry no cost semantics; real experiments cost weeks+$K vs CellFlow minutes |
| 3 | **Ingest Sanchis-Calleja 97 conditions as fidelity 0.8-0.9** | Sanchis-Calleja 2025 | HIGH | 3x the GP training set with real scRNA-seq data, currently only used for CellRank2/CellFlow |
| 4 | **Signature/target refinement via interpolation** | DeMeo 2025 | HIGH | Update Braun fetal brain target profile using Round 1 data; `target_profile` plumbing already exists |
| 5 | **GP prior warm-start across rounds** | Narayanan 2025 | HIGH | Reuse hyperparameters/NUTS chains between rounds instead of cold-starting each time |
| 6 | **Train CellFlow on our own data** | CellFlow 2025 | HIGH | Eliminate reliance on pre-trained model; use patterning screen + Amin/Kelley data we already have |
| 7 | **Duplicate experiments for QC/noise estimation** | BATCHIE 2025 | HIGH | Reserve 2-4 wells per round for replicates; calibrate GP noise and detect culture failures |

### High-Value Modeling Improvements

| # | Idea | Source Paper | Priority | Impact |
|---|---|---|---|---|
| 8 | **Additive + interaction kernel decomposition** | NAIAD 2025 | HIGH | Reduce effective parameters from O(d^2) to O(d); encodes morphogen independence prior |
| 9 | **Adaptive model complexity schedule** | NAIAD 2025 | HIGH | Shared-lengthscale GP in Round 1 -> ARD -> SAASBO as data grows |
| 10 | **Encode morphogen timing windows as GP dimensions** | Sanchis-Calleja 2025 | HIGH | Timing defines competence windows; our representation can't distinguish timing-only differences |
| 11 | **Per-cell-type GP models (MAP path)** | GPerturb 2025 | MEDIUM-HIGH | Per-output lengthscale matrix for interpretability; mirrors SAASBO ModelListGP pattern |
| 12 | **FBaxis_rank as regionalization target** | Sanchis-Calleja 2025 | MEDIUM-HIGH | Continuous A-P axis metric enables `--target-region` CLI for region-specific optimization |

### Defensive / Validation Ideas

| # | Idea | Source Paper | Priority | Impact |
|---|---|---|---|---|
| 13 | **Per-round fidelity monitoring** | McDonald 2025 | MEDIUM | Re-evaluate cross-fidelity correlation each round; auto-fallback to single-fidelity |
| 14 | **CellFlow saturation detection** | Cosenza 2022 | MEDIUM | Detect if CellFlow predictions plateau (like AlamarBlue saturation) |
| 15 | **Decoy robustness test** | McDonald 2025 | MEDIUM | Inject corrupted CellFlow predictions to measure GP resilience |
| 16 | **Convergence diagnostics** | Narayanan 2025 | MEDIUM | Track acquisition decay, posterior variance, recommendation clustering |
| 17 | **Ensemble disagreement diagnostic** | GPerturb 2025 | MEDIUM | Multi-restart fitting with recommendation stability scoring |

---

## Over-Engineering Warning

**Pillar 3 (multi-fidelity) is the most at-risk for over-engineering.** Per McDonald et al.:
- MF-BO only helps with weak-to-moderate, spatially heterogeneous fidelity correlation
- Strong correlation -> just use cheap fidelity (experimental funnel)
- No correlation -> low-fidelity is noise, single-fidelity GP is better
- **Validate before trusting the 3-tier system**

The single most impactful thing to do before Round 2 is **Idea #1: validate cross-fidelity correlation**. If CellFlow predictions are uncorrelated with wet-lab results, drop to 2-tier or single-fidelity.

---

## Implementation Phases

**Phase A (Pre-Round 2, ~1 week):**
Ideas 1, 5, 7 — validate fidelity correlation, warm-start GP, add replicate wells

**Phase B (Round 2 prep, ~2 weeks):**
Ideas 2, 3, 4, 6 — cost-aware acquisition, ingest patterning screen data, refine target, train CellFlow

**Phase C (Modeling improvements, ~3 weeks):**
Ideas 8, 9, 10, 11, 12 — kernel decomposition, adaptive complexity, timing encoding, per-cell-type models

**Phase D (Diagnostics, ongoing):**
Ideas 13-17 — monitoring, saturation detection, robustness, convergence, ensemble stability
