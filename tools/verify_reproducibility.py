"""Re-run the real pipeline and compare its metrics with the first run byte-for-byte."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.mlapp.pipeline import EXPORT_PATH, run_pipeline


def main() -> None:
    if not EXPORT_PATH.is_file():
        raise SystemExit("Run `python -m src.mlapp.pipeline` before this verification")

    reference_bytes = EXPORT_PATH.read_bytes()
    reference = json.loads(reference_bytes)

    with tempfile.TemporaryDirectory(prefix="jr05-repro-") as directory:
        root = Path(directory)
        rerun_path = root / "metrics.json"
        rerun = run_pipeline(models_dir=root / "models", export_path=rerun_path)
        rerun_bytes = rerun_path.read_bytes()

    if rerun != reference or rerun_bytes != reference_bytes:
        raise SystemExit("JR05 reproducibility check failed: repeated metrics differ")

    split = reference["split"]["row_counts"]
    selected = reference["selection"]["selected_model"]
    final_metrics = reference["final_test_metrics"]
    print(
        "JR05 reproducibility check passed: "
        f"split={split}, selected={selected}, final_test={final_metrics}"
    )


if __name__ == "__main__":
    main()
