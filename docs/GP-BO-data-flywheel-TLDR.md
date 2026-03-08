# GP + Bayesian Optimization Data Flywheel -- TLDR
**Created:** 2026-03-07

---

## What is a Gaussian Process (GP)?

A GP draws a smooth surface through your data points and tells you **how confident it is everywhere else**. Near data = confident. Far from data = uncertain.

- **Input**: morphogen concentrations (CHIR, BMP4, SHH, RA, etc.)
- **Output**: predicted cell type composition (% cortical, % hippocampal, % EC-like, etc.)
- **Bonus output**: uncertainty at every point -- "I predict 20% EC-like cells here, but I could be off by +/-12%"

**Fitting a GP** = giving it your experimental data so it learns (1) the smooth relationship between morphogens and cell fate, (2) how quickly outputs change as you vary each morphogen (lengthscale), and (3) how much noise exists.

**GP posterior** = the GP's updated belief after seeing data. Not a single curve -- a distribution over all plausible curves that pass through your data. Sample from it to get one plausible outcome.

---

## What is Bayesian Optimization (BO)?

BO uses the GP's uncertainty to pick the **next best experiment**. It balances:
- **Exploitation**: test near the current best condition
- **Exploration**: test where uncertainty is highest

The metric is **Expected Improvement (EI)**: which untested condition has the highest chance of beating your current best result?

---

## The Active Learning Loop

```
Round 0:  97 conditions from Sanchis-Calleja (free, published data)
          GP fits --> uncertainty heatmap
          BO picks 24 conditions that maximally reduce uncertainty
              |
Round 1:  Run 24 experiments in wet lab (~$15K with scRNA-seq)
          121 total conditions. GP re-fits. Uncertainty collapses.
          BO picks next 24 conditions (finer-grained, zooming in)
              |
Round 2:  145 conditions. Clear optimum emerging.
          BO picks next 24 (fine-tuning, adding new axes like RA timing)
              |
Round 3:  169 conditions. Sharp prediction. Protocol found.
```

**Total: ~72-96 conditions, ~$40-60K, 4-6 months.**
**vs brute force: thousands of conditions, years, millions of dollars.**

---

## What GP Gives You That Intuition Doesn't

1. **Quantified uncertainty** -- not "I think this might work" but "73% probability of >10% EC cells, 95% CI [3%, 22%]"
2. **Optimal experiment selection** -- BO mathematically picks conditions that give the most information per dollar
3. **Automatic importance ranking** -- GP learns short lengthscale = important factor, long = irrelevant. Free, no extra experiments.
4. **Multi-objective optimization** -- maximize EC fraction AND minimize cost AND maximize reproducibility (via BoTorch)
5. **Knows when to stop** -- when uncertainty is uniformly low, more experiments won't help

---

## The Engram Data Flywheel

```
Published data (free)
  --> GP --> Design Plate 1 --> Wet lab --> scRNA-seq
    --> GP update --> Design Plate 2 --> Wet lab --> scRNA-seq
      --> GP update --> Design Plate 3 --> ...
        --> Optimized protocol (3-4 rounds)
          --> Publication
            --> Start NEXT brain region (GP transfers knowledge)
```

Every brain region you solve adds data that helps predict the next one. The GP learns general morphogen-to-fate principles that transfer. By region #5, you may need 1-2 rounds instead of 4.

---

## Key Reference

**Sanchis-Calleja et al., 2025, Nature Methods** -- 97 morphogen conditions, 3 cell lines (WTC, H9, HES3), scRNA-seq at Day 21. Core training dataset.

GitHub: `github.com/quadbio/organoid_patterning_screen`

---

## Hackathon Demo (No Wet Lab Needed)

For the hackathon, simulate the loop: sample synthetic observations from the GP posterior (statistically plausible fake data), re-fit, show uncertainty collapsing in real-time. Demonstrates the process without needing actual wet lab results.
