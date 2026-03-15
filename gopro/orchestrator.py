"""Pipeline orchestrator for the GP-BO brain organoid pipeline.

Provides a single entry point to run the full pipeline (steps 00→06) or
any subset, using the dataset registry from ``datasets.yaml`` to determine
which datasets to process.

Usage (CLI)::

    python -m gopro.orchestrator                      # run all steps
    python -m gopro.orchestrator --steps 02 03 04     # run specific steps
    python -m gopro.orchestrator --from 02 --to 04    # run range
    python -m gopro.orchestrator --dataset amin_kelley # run one dataset
    python -m gopro.orchestrator --dry-run             # show plan only

Usage (Python)::

    from gopro.orchestrator import run_pipeline, PipelineConfig
    config = PipelineConfig(steps=["02", "03", "04"])
    run_pipeline(config)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from gopro.config import DATA_DIR, MODEL_DIR, get_logger
from gopro.datasets import (
    DatasetConfig,
    collect_fidelity_sources,
    get_dataset,
    get_real_datasets,
    get_virtual_datasets,
    load_dataset_registry,
)

logger = get_logger(__name__)

# Step definitions: id → (description, required inputs, produced outputs)
PIPELINE_STEPS = {
    "00":  "Download HNOCA + Braun from Zenodo",
    "00a": "Download GEO GSE233574 data",
    "00b": "Download patterning screen from Zenodo",
    "00c": "Build temporal atlas from patterning screen",
    "01":  "Convert GEO MTX to AnnData h5ad",
    "02":  "Map to HNOCA via scArches/scPoli",
    "03":  "Fidelity scoring vs Braun fetal brain",
    "04":  "GP-BO optimization loop",
    "05":  "CellRank 2 virtual data generation",
    "05v": "Generate visualization report",
    "06":  "CellFlow virtual protocol screening",
}

# Canonical ordering of steps
STEP_ORDER = ["00", "00a", "00b", "00c", "01", "02", "03", "04", "05", "06", "05v"]

# Step dependencies: step → list of prerequisite steps
STEP_DEPENDENCIES = {
    "01":  ["00a"],
    "02":  ["01", "00"],
    "03":  ["02", "00"],
    "04":  ["02"],
    "05":  ["02", "00c"],
    "06":  ["00c"],
    "05v": ["04"],
}


@dataclass
class StepResult:
    """Result of running a single pipeline step."""

    step: str
    success: bool
    message: str
    skipped: bool = False


@dataclass
class PipelineConfig:
    """Configuration for a pipeline run."""

    steps: Optional[list[str]] = None
    from_step: Optional[str] = None
    to_step: Optional[str] = None
    datasets: Optional[list[str]] = None
    dry_run: bool = False
    skip_validation: bool = False
    skip_download: bool = False

    def resolve_steps(self) -> list[str]:
        """Resolve step selection to an ordered list of step IDs."""
        if self.steps:
            # Explicit step list: validate and order
            for s in self.steps:
                if s not in PIPELINE_STEPS:
                    raise ValueError(
                        f"Unknown step '{s}'. Available: {list(PIPELINE_STEPS.keys())}"
                    )
            return [s for s in STEP_ORDER if s in self.steps]

        if self.from_step or self.to_step:
            start = self.from_step or STEP_ORDER[0]
            end = self.to_step or STEP_ORDER[-1]
            if start not in PIPELINE_STEPS:
                raise ValueError(f"Unknown --from step: '{start}'")
            if end not in PIPELINE_STEPS:
                raise ValueError(f"Unknown --to step: '{end}'")
            start_idx = STEP_ORDER.index(start)
            end_idx = STEP_ORDER.index(end)
            if start_idx > end_idx:
                raise ValueError(
                    f"--from {start} comes after --to {end} in pipeline order"
                )
            return STEP_ORDER[start_idx : end_idx + 1]

        # Default: all steps
        all_steps = list(STEP_ORDER)
        if self.skip_download:
            all_steps = [s for s in all_steps if not s.startswith("00")]
        return all_steps


def _check_step_inputs(step: str, datasets: list[DatasetConfig]) -> list[str]:
    """Check whether the inputs for a step exist on disk.

    Returns a list of missing input descriptions (empty if all present).
    """
    missing = []

    if step == "02":
        # Needs HNOCA reference + scPoli model + input h5ad per dataset
        ref = DATA_DIR / "hnoca_minimal_for_mapping.h5ad"
        if not ref.exists():
            missing.append(f"HNOCA reference: {ref}")
        if not MODEL_DIR.exists():
            missing.append(f"scPoli model: {MODEL_DIR}")
        for ds in datasets:
            if ds.input_path and not ds.input_path.exists():
                missing.append(f"Input h5ad for {ds.name}: {ds.input_path}")

    elif step == "03":
        braun = DATA_DIR / "braun-et-al_minimal_for_mapping.h5ad"
        if not braun.exists():
            missing.append(f"Braun fetal brain reference: {braun}")
        for ds in datasets:
            mapped = DATA_DIR / f"{ds.name}_mapped.h5ad"
            if not mapped.exists():
                missing.append(f"Mapped h5ad for {ds.name}: {mapped}")

    elif step == "04":
        for ds in datasets:
            if ds.fractions_path and not ds.fractions_path.exists():
                missing.append(f"Fractions CSV for {ds.name}: {ds.fractions_path}")
            if ds.morphogens_path and not ds.morphogens_path.exists():
                missing.append(f"Morphogens CSV for {ds.name}: {ds.morphogens_path}")

    elif step == "05":
        atlas = DATA_DIR / "azbukina_temporal_atlas.h5ad"
        if not atlas.exists():
            missing.append(f"Temporal atlas: {atlas}")

    return missing


def _validate_pre_step(step: str, datasets: list[DatasetConfig]) -> list[str]:
    """Run pre-step validation. Returns list of warnings."""
    warnings = []
    try:
        if step == "04":
            from gopro.validation import validate_training_csvs
            for ds in datasets:
                fp = ds.fractions_path
                mp = ds.morphogens_path
                if fp and mp and fp.exists() and mp.exists():
                    w = validate_training_csvs(fp, mp)
                    warnings.extend(w)
    except Exception as e:
        warnings.append(f"Pre-validation for step {step} failed: {e}")
    return warnings


def _validate_post_step(step: str, datasets: list[DatasetConfig]) -> list[str]:
    """Run post-step validation. Returns list of warnings."""
    warnings = []
    try:
        if step == "02":
            from gopro.validation import validate_mapped_adata
            for ds in datasets:
                mapped = DATA_DIR / f"{ds.name}_mapped.h5ad"
                if mapped.exists():
                    w = validate_mapped_adata(mapped, condition_key=ds.condition_key)
                    warnings.extend(w)

        elif step == "03":
            from gopro.validation import validate_fidelity_report
            report = DATA_DIR / "fidelity_report.csv"
            if report.exists():
                w = validate_fidelity_report(report)
                warnings.extend(w)

        elif step == "04":
            from gopro.validation import validate_training_csvs
            for ds in datasets:
                fp = ds.fractions_path
                mp = ds.morphogens_path
                if fp and mp and fp.exists() and mp.exists():
                    w = validate_training_csvs(fp, mp)
                    warnings.extend(w)
    except Exception as e:
        warnings.append(f"Post-validation for step {step} failed: {e}")
    return warnings


def build_execution_plan(config: PipelineConfig) -> list[tuple[str, str, list[str]]]:
    """Build the execution plan as a list of (step_id, description, missing_inputs).

    Returns:
        List of (step, description, missing_inputs) tuples.
    """
    steps = config.resolve_steps()
    registry = load_dataset_registry()

    # Resolve dataset filter
    if config.datasets:
        datasets = [get_dataset(name) for name in config.datasets]
    else:
        datasets = [ds for ds in registry.values() if ds.enabled]

    # Only consider real datasets for mapping/scoring steps
    real_datasets = [ds for ds in datasets if ds.fidelity == 1.0]

    plan = []
    for step in steps:
        desc = PIPELINE_STEPS.get(step, "Unknown step")
        step_datasets = real_datasets if step in ("02", "03") else datasets
        missing = _check_step_inputs(step, step_datasets)
        plan.append((step, desc, missing))

    return plan


def run_pipeline(config: PipelineConfig) -> list[StepResult]:
    """Run the pipeline according to the given configuration.

    Args:
        config: Pipeline configuration.

    Returns:
        List of StepResult objects, one per step.
    """
    steps = config.resolve_steps()
    registry = load_dataset_registry()

    # Resolve dataset filter
    if config.datasets:
        datasets = [get_dataset(name) for name in config.datasets]
    else:
        datasets = [ds for ds in registry.values() if ds.enabled]

    real_datasets = [ds for ds in datasets if ds.fidelity == 1.0]

    logger.info("Pipeline plan: %d steps, %d datasets", len(steps), len(datasets))
    for step in steps:
        logger.info("  Step %s: %s", step, PIPELINE_STEPS.get(step, "?"))
    for ds in datasets:
        logger.info("  Dataset: %s (fidelity=%.1f)", ds.name, ds.fidelity)

    if config.dry_run:
        plan = build_execution_plan(config)
        results = []
        for step, desc, missing in plan:
            if missing:
                msg = f"Would skip (missing: {', '.join(missing[:3])})"
                results.append(StepResult(step=step, success=True, message=msg, skipped=True))
            else:
                results.append(StepResult(step=step, success=True, message=f"Would run: {desc}"))
        return results

    results = []
    for step in steps:
        step_datasets = real_datasets if step in ("02", "03") else datasets

        # Pre-validation
        if not config.skip_validation:
            pre_warnings = _validate_pre_step(step, step_datasets)
            for w in pre_warnings:
                logger.warning("Pre-step %s: %s", step, w)

        # Check inputs
        missing = _check_step_inputs(step, step_datasets)
        if missing:
            msg = f"Skipped (missing inputs: {', '.join(missing[:3])})"
            logger.warning("Step %s: %s", step, msg)
            results.append(StepResult(step=step, success=True, message=msg, skipped=True))
            continue

        # Execute step
        logger.info("Running step %s: %s", step, PIPELINE_STEPS.get(step, "?"))
        try:
            _execute_step(step, step_datasets, config)
            msg = f"Completed: {PIPELINE_STEPS.get(step, '?')}"
            logger.info("Step %s: %s", step, msg)
            result = StepResult(step=step, success=True, message=msg)
        except Exception as e:
            msg = f"Failed: {e}"
            logger.error("Step %s: %s", step, msg)
            result = StepResult(step=step, success=False, message=msg)
            results.append(result)
            break  # Stop on first failure

        # Post-validation
        if not config.skip_validation:
            post_warnings = _validate_post_step(step, step_datasets)
            for w in post_warnings:
                logger.warning("Post-step %s: %s", step, w)

        results.append(result)

    n_completed = sum(1 for r in results if r.success and not r.skipped)
    n_skipped = sum(1 for r in results if r.skipped)
    n_failed = sum(1 for r in results if not r.success)
    logger.info(
        "Pipeline finished: %d completed, %d skipped, %d failed",
        n_completed, n_skipped, n_failed,
    )

    return results


def _execute_step(
    step: str,
    datasets: list[DatasetConfig],
    config: PipelineConfig,
) -> None:
    """Execute a single pipeline step.

    This function dispatches to the appropriate step module. Heavy
    imports happen inside each branch to avoid loading all dependencies
    at module import time.
    """
    if step == "00":
        from gopro import _load_cached
        mod = _load_cached("00_zenodo_download.py")
        mod.download_known_records()

    elif step == "00a":
        from gopro import _load_cached
        mod = _load_cached("00a_download_geo.py")
        mod.download_all()

    elif step == "00b":
        from gopro import _load_cached
        mod = _load_cached("00b_download_patterning_screen.py")
        mod.download_patterning_screen()

    elif step == "00c":
        from gopro import _load_cached
        mod = _load_cached("00c_build_temporal_atlas.py")
        mod.build_temporal_atlas()

    elif step == "01":
        from gopro import _load_cached
        mod = _load_cached("01_load_and_convert_data.py")
        mod.convert_all()

    elif step == "02":
        _run_mapping(datasets)

    elif step == "03":
        _run_fidelity_scoring(datasets)

    elif step == "04":
        _run_gpbo(datasets, config)

    elif step == "05":
        _run_cellrank2_virtual(datasets)

    elif step == "06":
        _run_cellflow_virtual()

    elif step == "05v":
        _run_visualization()

    else:
        raise ValueError(f"No executor for step '{step}'")


def _run_mapping(datasets: list[DatasetConfig]) -> None:
    """Run step 02 for each real dataset."""
    from gopro import run_mapping_pipeline

    ref_path = DATA_DIR / "hnoca_minimal_for_mapping.h5ad"

    for ds in datasets:
        if ds.fidelity != 1.0 or ds.input_path is None:
            continue
        if not ds.input_path.exists():
            logger.warning("Skipping %s: input not found at %s", ds.name, ds.input_path)
            continue

        logger.info("Mapping dataset: %s", ds.name)
        query, fractions, region_fractions = run_mapping_pipeline(
            query_path=ds.input_path,
            ref_path=ref_path,
            model_dir=MODEL_DIR,
            output_prefix=ds.name,
            condition_key=ds.condition_key,
            batch_key=ds.batch_key or "sample",
        )

        # Save outputs
        fractions.to_csv(str(DATA_DIR / ds.fractions_file))
        region_fractions.to_csv(
            str(DATA_DIR / f"gp_training_regions_{ds.name}.csv")
        )
        output_path = DATA_DIR / f"{ds.name}_mapped.h5ad"
        query.write(str(output_path), compression="gzip")
        logger.info("Saved mapped data for %s", ds.name)


def _run_fidelity_scoring(datasets: list[DatasetConfig]) -> None:
    """Run step 03 for each mapped dataset."""
    from gopro import run_fidelity_scoring

    for ds in datasets:
        if ds.fidelity != 1.0:
            continue
        mapped_path = DATA_DIR / f"{ds.name}_mapped.h5ad"
        if not mapped_path.exists():
            logger.warning("Skipping fidelity scoring for %s: no mapped data", ds.name)
            continue

        logger.info("Scoring fidelity for: %s", ds.name)
        run_fidelity_scoring(
            mapped_path=mapped_path,
            output_prefix=ds.name,
        )


def _run_gpbo(datasets: list[DatasetConfig], config: PipelineConfig) -> None:
    """Run step 04: GP-BO optimization."""
    from gopro import run_gpbo_loop

    # Find primary dataset for GP-BO
    real = [ds for ds in datasets if ds.fidelity == 1.0 and ds.has_training_data()]
    if not real:
        raise RuntimeError("No real datasets with training data found for GP-BO")

    primary = real[0]  # First real dataset is the primary

    # Collect virtual sources from all enabled datasets
    virtual_sources = []
    for ds in datasets:
        if ds is primary:
            continue
        src = ds.as_fidelity_source()
        if src is not None:
            virtual_sources.append(src)

    run_gpbo_loop(
        fractions_csv=primary.fractions_path,
        morphogen_csv=primary.morphogens_path,
        virtual_sources=virtual_sources if virtual_sources else None,
    )


def _run_cellrank2_virtual(datasets: list[DatasetConfig]) -> None:
    """Run step 05: CellRank 2 virtual data generation."""
    from gopro import generate_virtual_training_data

    atlas_path = DATA_DIR / "azbukina_temporal_atlas.h5ad"
    if not atlas_path.exists():
        raise RuntimeError(f"Temporal atlas not found: {atlas_path}")

    for ds in datasets:
        if ds.fidelity != 1.0:
            continue
        mapped_path = DATA_DIR / f"{ds.name}_mapped.h5ad"
        if not mapped_path.exists():
            continue

        logger.info("Generating CellRank 2 virtual data for: %s", ds.name)
        generate_virtual_training_data(
            atlas_path=atlas_path,
            query_path=mapped_path,
        )


def _run_cellflow_virtual() -> None:
    """Run step 06: CellFlow virtual screening."""
    from gopro import _load_cached
    mod = _load_cached("06_cellflow_virtual.py")
    if hasattr(mod, "run_virtual_screen"):
        mod.run_virtual_screen()
    else:
        logger.warning("CellFlow run_virtual_screen() not available")


def _run_visualization() -> None:
    """Run step 05v: Generate visualization report."""
    from gopro import _load_cached
    mod = _load_cached("05_visualize.py")
    if hasattr(mod, "main"):
        mod.main()
    else:
        logger.warning("Visualization main() not available")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="GP-BO Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Steps: " + ", ".join(
            f"{k} ({v})" for k, v in PIPELINE_STEPS.items()
        ),
    )
    parser.add_argument(
        "--steps", nargs="+", default=None,
        help="Specific steps to run (e.g., 02 03 04)",
    )
    parser.add_argument(
        "--from", dest="from_step", default=None,
        help="First step to run (inclusive)",
    )
    parser.add_argument(
        "--to", dest="to_step", default=None,
        help="Last step to run (inclusive)",
    )
    parser.add_argument(
        "--dataset", nargs="+", dest="datasets", default=None,
        help="Specific dataset names to process",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show execution plan without running",
    )
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Skip pre/post-step validation checks",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip download steps (00, 00a, 00b)",
    )
    parser.add_argument(
        "--list-steps", action="store_true",
        help="List available steps and exit",
    )
    args = parser.parse_args()

    if args.list_steps:
        print("\nAvailable pipeline steps:")
        print("=" * 60)
        for step_id in STEP_ORDER:
            deps = STEP_DEPENDENCIES.get(step_id, [])
            dep_str = f" (depends on: {', '.join(deps)})" if deps else ""
            print(f"  {step_id:5s} {PIPELINE_STEPS[step_id]}{dep_str}")
        return

    config = PipelineConfig(
        steps=args.steps,
        from_step=args.from_step,
        to_step=args.to_step,
        datasets=args.datasets,
        dry_run=args.dry_run,
        skip_validation=args.skip_validation,
        skip_download=args.skip_download,
    )

    results = run_pipeline(config)

    if args.dry_run:
        print("\nDry-run execution plan:")
        print("=" * 60)
        for r in results:
            status = "SKIP" if r.skipped else "RUN"
            print(f"  [{status}] Step {r.step}: {r.message}")
    else:
        print("\nPipeline results:")
        print("=" * 60)
        for r in results:
            status = "SKIP" if r.skipped else ("OK" if r.success else "FAIL")
            print(f"  [{status}] Step {r.step}: {r.message}")


if __name__ == "__main__":
    main()
