"""Accept a 4090 environment check after three real updates and checkpoint I/O."""

import argparse
import json
import math
from pathlib import Path

from verify_run import checkpoint_state


def verify(job: Path) -> dict:
    run = job / "train"
    records = [json.loads(line) for line in (run / "logs/metrics.jsonl").read_text().splitlines()]
    train = [row for row in records if row["event"] == "train"]
    if [row["optimizer_updates"] for row in train] != [1, 2, 3]:
        raise ValueError("Expected exactly three real parameter updates")
    for row in train:
        if not all(
            isinstance(value, (int, float)) and math.isfinite(value) for value in row["metrics"].values()
        ):
            raise ValueError("Non-finite or invalid training metrics")
    if not any(row["metrics"].get("grad_norm", 0) > 0 for row in train):
        raise ValueError("No non-zero gradient was observed")
    groups = json.loads((run / "logs/trainable_parameters.json").read_text())
    if groups["backbone"]["trainable"] != 0 or groups["action_head"]["trainable"] <= 0:
        raise ValueError("Unexpected trainable parameter groups")
    if not list((run / "logs/tensorboard").glob("events.out.tfevents.*")):
        raise FileNotFoundError("TensorBoard events missing")
    environment = json.loads((job / "environment.json").read_text())
    if "4090" not in environment["gpu"] or environment["profile"] != "4090-environment":
        raise ValueError("This report requires the 4090 validation profile")
    return {
        "status": "passed",
        "validation_kind": "4090_environment",
        "optimizer_updates": 3,
        "checkpoint": checkpoint_state(run, 3),
        "parameter_groups": groups,
        "gpu": environment["gpu"],
        "metrics": [row["metrics"] for row in train],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.job_root)
    (args.job_root / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
