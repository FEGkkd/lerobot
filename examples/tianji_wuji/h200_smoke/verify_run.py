"""Accept a smoke run only after 50 updates, a real restart, and 50 more updates."""

import argparse
import json
from pathlib import Path

import numpy as np


def checkpoint_state(run: Path, step: int) -> dict:
    from safetensors import safe_open

    checkpoint = run / "checkpoints" / f"{step:06d}"
    state_dir = checkpoint / "training_state"
    state = json.loads((state_dir / "training_step.json").read_text())
    if state["step"] != step or state["grad_accum_steps"] != 1 or state["dp_world_size"] != 1:
        raise ValueError(f"Unexpected checkpoint metadata at {step}")
    scheduler = json.loads((state_dir / "scheduler_state.json").read_text())
    if scheduler["last_epoch"] != step:
        raise ValueError("Scheduler state did not advance with the optimizer.")
    for name in ("optimizer_param_groups.json", "rng_state.safetensors"):
        if not (state_dir / name).is_file():
            raise FileNotFoundError(state_dir / name)
    with safe_open(state_dir / "optimizer_state.safetensors", framework="np") as tensors:
        updates = [
            float(tensors.get_tensor(key).item()) for key in list(tensors.keys()) if key.endswith("/step")
        ]
    if not updates or max(updates) != step:
        raise ValueError(f"AdamW state was not restored/updated through step {step}: {updates[:5]}")
    saved_config = json.loads((checkpoint / "pretrained_model/train_config.json").read_text())
    if saved_config["steps"] != 100:
        raise ValueError("The planned 100-step scheduler horizon changed.")
    return {"step": step, "optimizer_step_max": max(updates), "scheduler_step": scheduler["last_epoch"]}


def compare_weights(run: Path) -> dict:
    from safetensors import safe_open

    paths = [run / "checkpoints" / f"{step:06d}" / "pretrained_model/model.safetensors" for step in (50, 100)]
    with safe_open(paths[0], framework="np") as first, safe_open(paths[1], framework="np") as second:
        keys = list(first.keys())
        head = next(key for key in keys if "action_head" in key and key.endswith("vlln.weight"))
        frozen = next(key for key in keys if "backbone" in key and "norm" in key and key.endswith("weight"))
        if np.array_equal(first.get_tensor(head), second.get_tensor(head)):
            raise ValueError("The action-head sample weight did not change after restart.")
        if not np.array_equal(first.get_tensor(frozen), second.get_tensor(frozen)):
            raise ValueError("A frozen backbone sample weight changed.")
    return {"updated_tensor": head, "unchanged_backbone_tensor": frozen}


def verify(job: Path) -> dict:
    run = job / "train"
    records = [json.loads(line) for line in (run / "logs/metrics.jsonl").read_text().splitlines()]
    steps = [row["optimizer_updates"] for row in records if row["event"] == "train"]
    if steps != list(range(1, 101)):
        raise ValueError("Expected exactly updates 1..100, with no repeated/lost steps after restart.")
    validations = [row["optimizer_updates"] for row in records if row["event"] == "validation"]
    if validations != [25, 50, 75, 100]:
        raise ValueError("Missing validation metrics.")
    for row in records:
        if not all(np.isfinite(value) for value in row["metrics"].values()):
            raise ValueError("Non-finite metrics.")
    groups = json.loads((run / "logs/trainable_parameters.json").read_text())
    if groups["backbone"]["trainable"] != 0 or groups["action_head"]["trainable"] <= 0:
        raise ValueError("Unexpected partial-finetuning parameter groups.")
    if not list((run / "logs/tensorboard").glob("events.out.tfevents.*")):
        raise FileNotFoundError("TensorBoard events were not written.")
    results = [checkpoint_state(run, step) for step in (50, 100)]
    return {
        "status": "passed",
        "optimizer_updates": 100,
        "checkpoints": results,
        "validation_updates": validations,
        "parameter_groups": groups,
        "weight_comparison": compare_weights(run),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.job_root)
    (args.job_root / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
