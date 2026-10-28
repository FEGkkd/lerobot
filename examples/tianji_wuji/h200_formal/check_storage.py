"""Check shared-disk space before creating seven full training checkpoints."""

import argparse
import json
import shutil
from pathlib import Path


def check(job: Path) -> dict:
    job = job.resolve()
    if not str(job).startswith("/inspire/"):
        raise ValueError("Formal outputs must use the persistent /inspire shared disk")
    usage = shutil.disk_usage(job)
    minimum = 220 * 1024**3
    if usage.free < minimum:
        raise RuntimeError(f"Need220GiB free for checkpoints, only {usage.free / 1024**3:.1f}GiB available")
    report = {
        "status": "passed",
        "free_gib": usage.free / 1024**3,
        "minimum_free_gib": 220,
        "estimated_checkpoints_decimal_gb": 179,
        "note": "Filesystem free space does not establish remaining per-user/project quota",
    }
    (job / "storage.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.job_root)))
