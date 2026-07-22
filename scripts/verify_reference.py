#!/usr/bin/env python3
"""Verify the committed EXP-0004 reference result package."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = REPO_ROOT / "reports" / "assets" / "EXP-0004"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_reference(root: Path = REFERENCE_ROOT) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = root / item["path"]
        actual = sha256_file(path)
        if actual != item["sha256"]:
            raise ValueError(f"SHA256 mismatch for {path}: {actual}")

    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((root / "evaluations.csv").open(encoding="utf-8")))
    independent = json.loads((root / "independent_final_eval.json").read_text(encoding="utf-8"))
    expected = manifest["expected"]
    checks = {
        "evaluation_count": len(rows) == expected["evaluation_count"],
        "final_mean": math.isclose(
            float(summary["final_mean_episode_return"]), expected["final_mean_episode_return"], abs_tol=1e-12
        ),
        "reference_score": math.isclose(float(summary["reference_score"]), expected["reference_score"], abs_tol=1e-12),
        "independent_exact_match": independent["episode_returns_exact_match"] is True,
        "checkpoint_sha256": independent["checkpoint_sha256"] == expected["checkpoint_sha256"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"reference checks failed: {', '.join(failed)}")
    return {"status": "ok", "root": str(root), **expected}


def main() -> None:
    print(json.dumps(verify_reference(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

