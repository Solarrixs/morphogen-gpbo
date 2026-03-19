"""Noise-robustness pre-screening for GP-BO settings.

Runs the ToyMorphogenFunction at varying noise levels and batch sizes
to determine robust BO configurations before committing to expensive
wet-lab experiments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from gopro.config import get_logger, MORPHOGEN_COLUMNS
from gopro.benchmarks.toy_morphogen_function import ToyMorphogenFunction

logger = get_logger(__name__)


def run_noise_sweep(
    noise_levels: tuple[float, ...] = (0.01, 0.05, 0.1, 0.2),
    batch_sizes: tuple[int, ...] = (8, 16, 24),
    n_rounds: int = 3,
    n_initial: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Run parametric noise x batch size sweep on the toy function.

    For each (noise_level, batch_size) combination, simulates n_rounds
    of BO and records the best-found objective value (mean of Neuron
    fraction across the batch).

    This does NOT run actual GP fitting (too slow for a sweep). Instead
    it uses random search as a baseline to characterize how noise degrades
    optimization quality at each batch size.

    Args:
        noise_levels: Gaussian noise std in logit space.
        batch_sizes: Number of candidates per round.
        n_rounds: Number of optimization rounds to simulate.
        n_initial: Initial random samples before optimization.
        seed: Random seed.

    Returns:
        DataFrame with columns: noise_level, batch_size, round,
        best_observed, mean_observed, n_evaluated.
    """
    n_morphogens = len(MORPHOGEN_COLUMNS)
    records: list[dict] = []

    for noise in noise_levels:
        fn = ToyMorphogenFunction(noise_std=noise, seed=seed)
        for batch_size in batch_sizes:
            rng = np.random.default_rng(seed)

            # Generate initial random points
            x_init = rng.random((n_initial, n_morphogens)) * 10.0
            y_init = fn.evaluate(x_init)
            # Neuron fraction is column 0
            neuron_fracs = y_init[:, 0].tolist()

            best_so_far = max(neuron_fracs)

            records.append(
                {
                    "noise_level": noise,
                    "batch_size": batch_size,
                    "round": 0,
                    "best_observed": best_so_far,
                    "mean_observed": float(np.mean(neuron_fracs)),
                    "n_evaluated": n_initial,
                }
            )

            total_evaluated = n_initial

            for rnd in range(1, n_rounds + 1):
                x_batch = rng.random((batch_size, n_morphogens)) * 10.0
                y_batch = fn.evaluate(x_batch)
                batch_neuron = y_batch[:, 0].tolist()
                neuron_fracs.extend(batch_neuron)
                total_evaluated += batch_size

                best_so_far = max(best_so_far, max(batch_neuron))

                records.append(
                    {
                        "noise_level": noise,
                        "batch_size": batch_size,
                        "round": rnd,
                        "best_observed": best_so_far,
                        "mean_observed": float(np.mean(batch_neuron)),
                        "n_evaluated": total_evaluated,
                    }
                )

            logger.debug(
                "noise=%.3f batch=%d: best_observed=%.4f after %d evals",
                noise,
                batch_size,
                best_so_far,
                total_evaluated,
            )

    return pd.DataFrame(records)


def summarize_noise_sweep(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize sweep results with regret and robustness assessment.

    Groups by (noise_level, batch_size), takes the final round's
    best_observed, computes regret relative to the noiseless optimum,
    and labels each configuration as 'robust' if regret < 0.1.

    Args:
        results: Output of :func:`run_noise_sweep`.

    Returns:
        DataFrame with columns: noise_level, batch_size, final_best,
        regret, recommendation.
    """
    # Compute noiseless optimum: evaluate toy function at its own optimum
    fn_clean = ToyMorphogenFunction(noise_std=0.0, seed=42)
    optimum_x = fn_clean.optimum.reshape(1, -1)
    optimum_neuron = float(fn_clean.evaluate(optimum_x)[0, 0])

    # Get final round per (noise_level, batch_size)
    idx = results.groupby(["noise_level", "batch_size"])["round"].idxmax()
    final = results.loc[idx].copy()

    final["regret"] = optimum_neuron - final["best_observed"]
    # Clamp negative regret (random search could exceed optimum with noise)
    final["regret"] = final["regret"].clip(lower=0.0)
    final["recommendation"] = final["regret"].apply(
        lambda r: "robust" if r < 0.1 else "sensitive"
    )

    summary = final[
        ["noise_level", "batch_size", "best_observed", "regret", "recommendation"]
    ].rename(columns={"best_observed": "final_best"})

    return summary.reset_index(drop=True)
