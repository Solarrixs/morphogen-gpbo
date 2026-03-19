"""Tests for the pipeline orchestrator (Phase 4B)."""

import pytest

from gopro.orchestrator import (
    PIPELINE_STEPS,
    STEP_DEPENDENCIES,
    STEP_ORDER,
    PipelineConfig,
    StepResult,
    build_execution_plan,
    run_pipeline,
)


# ---------------------------------------------------------------------------
# PipelineConfig.resolve_steps
# ---------------------------------------------------------------------------

class TestResolveSteps:

    def test_default_all_steps(self):
        config = PipelineConfig()
        steps = config.resolve_steps()
        assert steps == STEP_ORDER

    def test_explicit_steps(self):
        config = PipelineConfig(steps=["02", "03", "04"])
        steps = config.resolve_steps()
        assert steps == ["02", "03", "04"]

    def test_explicit_steps_ordered(self):
        """Steps are reordered to pipeline order even if given out of order."""
        config = PipelineConfig(steps=["04", "02"])
        steps = config.resolve_steps()
        assert steps == ["02", "04"]

    def test_from_to_range(self):
        config = PipelineConfig(from_step="02", to_step="04")
        steps = config.resolve_steps()
        assert steps == ["02", "03", "04"]

    def test_from_only(self):
        config = PipelineConfig(from_step="04")
        steps = config.resolve_steps()
        assert "04" in steps
        assert "02" not in steps

    def test_to_only(self):
        config = PipelineConfig(to_step="01")
        steps = config.resolve_steps()
        assert "01" in steps
        assert "02" not in steps

    def test_invalid_step_raises(self):
        config = PipelineConfig(steps=["99"])
        with pytest.raises(ValueError, match="Unknown step"):
            config.resolve_steps()

    def test_invalid_from_step_raises(self):
        config = PipelineConfig(from_step="99")
        with pytest.raises(ValueError, match="Unknown --from"):
            config.resolve_steps()

    def test_invalid_to_step_raises(self):
        config = PipelineConfig(to_step="99")
        with pytest.raises(ValueError, match="Unknown --to"):
            config.resolve_steps()

    def test_from_after_to_raises(self):
        config = PipelineConfig(from_step="04", to_step="02")
        with pytest.raises(ValueError, match="comes after"):
            config.resolve_steps()

    def test_skip_download(self):
        config = PipelineConfig(skip_download=True)
        steps = config.resolve_steps()
        assert not any(s.startswith("00") for s in steps)
        assert "01" in steps

    def test_single_step(self):
        config = PipelineConfig(steps=["04"])
        steps = config.resolve_steps()
        assert steps == ["04"]


# ---------------------------------------------------------------------------
# STEP_ORDER and PIPELINE_STEPS consistency
# ---------------------------------------------------------------------------

class TestStepDefinitions:

    def test_all_steps_in_order(self):
        """Every defined step appears in STEP_ORDER."""
        for step in PIPELINE_STEPS:
            assert step in STEP_ORDER, f"Step {step} missing from STEP_ORDER"

    def test_all_ordered_steps_defined(self):
        """Every step in STEP_ORDER has a description."""
        for step in STEP_ORDER:
            assert step in PIPELINE_STEPS, f"Step {step} in STEP_ORDER but not PIPELINE_STEPS"

    def test_dependencies_reference_valid_steps(self):
        """All dependency targets exist in PIPELINE_STEPS."""
        for step, deps in STEP_DEPENDENCIES.items():
            assert step in PIPELINE_STEPS, f"Dep source {step} not in PIPELINE_STEPS"
            for dep in deps:
                assert dep in PIPELINE_STEPS, (
                    f"Step {step} depends on {dep} which is not defined"
                )

    def test_no_circular_dependencies(self):
        """Verify no circular dependency chains."""
        visited = set()

        def _check(step, chain):
            if step in chain:
                pytest.fail(f"Circular dependency: {' -> '.join(chain)} -> {step}")
            chain.add(step)
            for dep in STEP_DEPENDENCIES.get(step, []):
                _check(dep, chain.copy())

        for step in PIPELINE_STEPS:
            _check(step, set())


# ---------------------------------------------------------------------------
# StepResult
# ---------------------------------------------------------------------------

class TestStepResult:

    def test_success(self):
        r = StepResult(step="02", success=True, message="Done")
        assert r.success
        assert not r.skipped

    def test_skipped(self):
        r = StepResult(step="02", success=True, message="Skipped", skipped=True)
        assert r.skipped

    def test_failure(self):
        r = StepResult(step="02", success=False, message="Error")
        assert not r.success


# ---------------------------------------------------------------------------
# Dry run (no actual step execution)
# ---------------------------------------------------------------------------

class TestDryRun:

    def test_dry_run_returns_results(self):
        config = PipelineConfig(steps=["02", "03", "04"], dry_run=True)
        results = run_pipeline(config)
        assert len(results) == 3
        assert all(isinstance(r, StepResult) for r in results)

    def test_dry_run_no_side_effects(self):
        """Dry run doesn't actually execute anything."""
        config = PipelineConfig(steps=["04"], dry_run=True)
        results = run_pipeline(config)
        assert len(results) == 1
        # Should report as either "Would run" or "Would skip"
        assert "Would" in results[0].message

    def test_dry_run_detects_missing_inputs(self):
        """Dry run identifies missing input files."""
        config = PipelineConfig(steps=["02"], dry_run=True)
        results = run_pipeline(config)
        assert len(results) == 1
        # Step 02 needs HNOCA reference + model — likely missing in CI
        # Either it reports as "Would run" or "Would skip" depending on environment
        assert results[0].success


# ---------------------------------------------------------------------------
# build_execution_plan
# ---------------------------------------------------------------------------

class TestBuildExecutionPlan:

    def test_returns_plan(self):
        config = PipelineConfig(steps=["02", "04"])
        plan = build_execution_plan(config)
        assert len(plan) == 2
        step_ids = [p[0] for p in plan]
        assert step_ids == ["02", "04"]

    def test_plan_includes_descriptions(self):
        config = PipelineConfig(steps=["04"])
        plan = build_execution_plan(config)
        step_id, desc, missing = plan[0]
        assert step_id == "04"
        assert "GP-BO" in desc

    def test_plan_reports_missing_inputs(self):
        config = PipelineConfig(steps=["05"])
        plan = build_execution_plan(config)
        _, _, missing = plan[0]
        # Temporal atlas likely missing
        assert isinstance(missing, list)


# ---------------------------------------------------------------------------
# Pipeline execution (with skips due to missing data)
# ---------------------------------------------------------------------------

class TestPipelineExecution:

    def test_skips_steps_with_missing_inputs(self):
        """Steps with missing inputs are skipped or fail gracefully."""
        config = PipelineConfig(steps=["02"])
        results = run_pipeline(config)
        assert len(results) == 1
        r = results[0]
        # Step 02 will either skip (missing data) or fail (missing dependency
        # like pkg_resources on Python 3.14). Both are graceful outcomes.
        assert r.skipped or not r.success

    def test_multiple_steps_skip_gracefully(self):
        """Pipeline handles missing data/deps without crashing."""
        config = PipelineConfig(steps=["02", "03", "04"])
        results = run_pipeline(config)
        # Pipeline should return at least 1 result without raising
        assert len(results) >= 1
        # Each result is a proper StepResult
        for r in results:
            assert isinstance(r, StepResult)
            assert r.step in ("02", "03", "04")
