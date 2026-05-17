"""Tests for CLI argument parsing — no GPU, uses sys.argv mocking.

sklearn is mocked at module import time so tests run without it installed.
"""
from __future__ import annotations

import sys
import types
import importlib
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Stub out sklearn before warp_score.cli (and its transitive imports) loads,
# so the evaluator module-level `from sklearn.metrics import ...` succeeds.
# ---------------------------------------------------------------------------

def _stub_sklearn() -> None:
    """Insert minimal sklearn stubs into sys.modules if not already present."""
    if "sklearn" in sys.modules:
        return
    sklearn_mod = types.ModuleType("sklearn")
    metrics_mod = types.ModuleType("sklearn.metrics")
    for fn_name in (
        "roc_auc_score",
        "average_precision_score",
        "precision_recall_curve",
        "roc_curve",
    ):
        setattr(metrics_mod, fn_name, MagicMock(return_value=0.0))
    sklearn_mod.metrics = metrics_mod
    sys.modules["sklearn"] = sklearn_mod
    sys.modules["sklearn.metrics"] = metrics_mod


_stub_sklearn()

# Now it is safe to import the CLI
from warp_score.cli import main  # noqa: E402


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_calibrate_subcommand_parses():
    """'calibrate' subcommand is recognised by argparse; any subsequent error is
    FileNotFoundError or RuntimeError (not argparse SystemExit(2))."""
    try:
        main(["calibrate"])
    except (FileNotFoundError, RuntimeError, Exception) as exc:
        # Argparse errors raise SystemExit(2); anything else is fine
        if isinstance(exc, SystemExit):
            assert exc.code != 2, (
                f"argparse rejected 'calibrate' with SystemExit(2): {exc}"
            )


def test_detect_requires_calib_or_artifacts():
    """detect subcommand with a nonexistent artifacts dir raises FileNotFoundError
    (calibration.npz not found) rather than an argparse error."""
    with pytest.raises(FileNotFoundError):
        main(["--artifacts_dir", "/tmp/_nonexistent_warptest_9999", "detect"])


def test_eval_requires_labels_arg():
    """'eval' without --labels should exit with argparse code 2."""
    with pytest.raises(SystemExit) as exc_info:
        main(["eval"])
    assert exc_info.value.code == 2, (
        f"Expected SystemExit(2) from argparse, got {exc_info.value.code}"
    )


def test_global_ref_dir_flag_accepted():
    """--ref_dir is a valid global flag; any non-argparse error (e.g.
    FileNotFoundError) is acceptable — SystemExit(2) is not."""
    try:
        main(["--ref_dir", "/tmp", "calibrate"])
    except SystemExit as exc:
        assert exc.code != 2, (
            f"argparse rejected --ref_dir with SystemExit(2): {exc}"
        )
    except Exception:
        # FileNotFoundError, RuntimeError, ImportError etc. are all fine
        pass


def test_global_query_dirs_flags_accepted():
    """--query_high_dir and --query_low_dir are valid global flags."""
    try:
        main([
            "--query_high_dir", "/tmp",
            "--query_low_dir", "/tmp",
            "calibrate",
        ])
    except SystemExit as exc:
        assert exc.code != 2, (
            f"argparse rejected --query_*_dir with SystemExit(2): {exc}"
        )
    except Exception:
        pass
