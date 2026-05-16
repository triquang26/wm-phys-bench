"""Unit tests for TaskCalibration and CalibrationArtifact save/load."""
import numpy as np
import pytest

from warp_score.calibrator import CalibrationArtifact, TaskCalibration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(name: str, n: int = 5, seed: int = 0) -> TaskCalibration:
    rng = np.random.default_rng(seed)
    return TaskCalibration(
        task=name,
        n_refs=n,
        ivar_dist=np.sort(rng.uniform(0, 1, n).astype(np.float32)),
        peak_dist=np.sort(rng.uniform(0, 5, n).astype(np.float32)),
        cert_dist=np.sort(rng.uniform(0, 1, n).astype(np.float32)),
    )


def _make_artifact(tasks: dict[str, TaskCalibration]) -> CalibrationArtifact:
    all_ivar = np.concatenate([t.ivar_dist for t in tasks.values()])
    all_peak = np.concatenate([t.peak_dist for t in tasks.values()])
    all_cert = np.concatenate([t.cert_dist for t in tasks.values()])
    global_ = TaskCalibration(
        task="__global__",
        n_refs=len(all_ivar),
        ivar_dist=np.sort(all_ivar),
        peak_dist=np.sort(all_peak),
        cert_dist=np.sort(all_cert),
    )
    return CalibrationArtifact(
        tasks=tasks,
        global_=global_,
        config_snapshot={"vis_size": 224},
        created_at="2025-01-01T00:00:00",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_task_calibration_properties():
    """mean_ivar and std_ivar are computed correctly from ivar_dist=[1,2,3]."""
    tc = TaskCalibration(
        task="demo",
        n_refs=3,
        ivar_dist=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        peak_dist=np.array([0.0, 0.5, 1.0], dtype=np.float32),
        cert_dist=np.array([0.1, 0.5, 0.9], dtype=np.float32),
    )
    assert tc.mean_ivar == pytest.approx(2.0, abs=1e-5)
    # std with ddof=1 of [1,2,3] == 1.0
    assert tc.std_ivar == pytest.approx(1.0, abs=1e-5)


def test_calibration_artifact_save_load_roundtrip(tmp_path):
    """Two-task artifact can be saved and reloaded; distributions are identical."""
    tasks = {
        "task_a": _make_task("task_a", n=5, seed=1),
        "task_b": _make_task("task_b", n=7, seed=2),
    }
    artifact = _make_artifact(tasks)
    save_path = tmp_path / "calib.npz"
    artifact.save(save_path)

    loaded = CalibrationArtifact.load(save_path)

    assert set(loaded.tasks.keys()) == {"task_a", "task_b"}
    np.testing.assert_array_almost_equal(
        loaded.tasks["task_a"].ivar_dist, tasks["task_a"].ivar_dist,
    )
    np.testing.assert_array_almost_equal(
        loaded.tasks["task_b"].peak_dist, tasks["task_b"].peak_dist,
    )
    np.testing.assert_array_almost_equal(
        loaded.tasks["task_a"].cert_dist, tasks["task_a"].cert_dist,
    )


def test_calibration_artifact_save_load_global(tmp_path):
    """Global TaskCalibration is correctly round-tripped."""
    tasks = {
        "task_a": _make_task("task_a", n=5, seed=10),
        "task_b": _make_task("task_b", n=5, seed=11),
    }
    artifact = _make_artifact(tasks)
    save_path = tmp_path / "calib_global.npz"
    artifact.save(save_path)

    loaded = CalibrationArtifact.load(save_path)

    np.testing.assert_array_almost_equal(
        loaded.global_.ivar_dist, artifact.global_.ivar_dist,
    )
    np.testing.assert_array_almost_equal(
        loaded.global_.cert_dist, artifact.global_.cert_dist,
    )
    assert loaded.global_.task == "__global__"


def test_calibration_artifact_with_per_pixel(tmp_path):
    """per_pixel_var (N, H, W) array is preserved after save/load."""
    rng = np.random.default_rng(42)
    N, H, W = 5, 8, 8
    tc = TaskCalibration(
        task="pp_task",
        n_refs=N,
        ivar_dist=np.sort(rng.uniform(0, 1, N).astype(np.float32)),
        peak_dist=np.sort(rng.uniform(0, 5, N).astype(np.float32)),
        cert_dist=np.sort(rng.uniform(0, 1, N).astype(np.float32)),
        per_pixel_var=rng.uniform(0, 1, (N, H, W)).astype(np.float32),
    )
    artifact = _make_artifact({"pp_task": tc})
    save_path = tmp_path / "calib_pp.npz"
    artifact.save(save_path)

    loaded = CalibrationArtifact.load(save_path)

    assert loaded.tasks["pp_task"].per_pixel_var is not None
    assert loaded.tasks["pp_task"].per_pixel_var.shape == (N, H, W)
    np.testing.assert_array_almost_equal(
        loaded.tasks["pp_task"].per_pixel_var, tc.per_pixel_var,
    )


def test_n_tasks_correct():
    """n_tasks attribute equals the number of entries in tasks dict."""
    tasks = {
        "t1": _make_task("t1"),
        "t2": _make_task("t2"),
        "t3": _make_task("t3"),
    }
    artifact = _make_artifact(tasks)
    assert artifact.n_tasks == 3
