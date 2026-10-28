"""Verify the single-H200 baseline and select the best saved validation checkpoint."""

import argparse
import json
import math
import statistics
from datetime import datetime
from pathlib import Path

import numpy as np


def checkpoint_state(run: Path, step: int, horizon: int, batch_size: int) -> dict:
    from safetensors import safe_open

    checkpoint = run / "checkpoints" / f"{step:06d}"
    state_dir = checkpoint / "training_state"
    state = json.loads((state_dir / "training_step.json").read_text())
    if (state["step"], state["grad_accum_steps"], state["dp_world_size"]) != (step, 1, 1):
        raise ValueError(f"Unexpected checkpoint metadata at {step}")
    if state.get("batch_size") != batch_size:
        raise ValueError("Checkpoint batch size differs from the formal recipe")
    scheduler = json.loads((state_dir / "scheduler_state.json").read_text())
    if scheduler["last_epoch"] != step:
        raise ValueError("Scheduler count differs from the optimizer update count")
    for name in ("optimizer_param_groups.json", "rng_state.safetensors"):
        if not (state_dir / name).is_file():
            raise FileNotFoundError(state_dir / name)
    with safe_open(state_dir / "optimizer_state.safetensors", framework="np") as tensors:
        updates = [
            float(tensors.get_tensor(key).item()) for key in list(tensors.keys()) if key.endswith("/step")
        ]
    if not updates or max(updates) != step:
        raise ValueError(f"AdamW state was not restored through update {step}")
    config = json.loads((checkpoint / "pretrained_model/train_config.json").read_text())
    if config["steps"] != horizon or config["scheduler"]["num_warmup_steps"] != 1500:
        raise ValueError("The training horizon/warmup changed across a restart")
    model = checkpoint / "pretrained_model/model.safetensors"
    if not model.is_file():
        raise FileNotFoundError(model)
    return {"step": step, "optimizer_step": max(updates), "scheduler_step": scheduler["last_epoch"]}


def verify(job: Path, stop: int) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    config = json.loads((job / "train_config.json").read_text())
    horizon, batch = config["steps"], config["batch_size"]
    if (
        horizon != 30_000
        or batch != 4
        or config["accelerator"]["gradient_accumulation"]["steps"] != 1
        or config["max_eval_samples"] != 0
    ):
        raise ValueError("Unexpected formal update/sample budget")
    if stop not in (1000, horizon):
        raise ValueError("Expected either the 1000-update startup check or the full run")
    run = job / "train"
    rows = [json.loads(line) for line in (run / "logs/metrics.jsonl").read_text().splitlines()]
    train = [row for row in rows if row["event"] == "train"]
    validations = [row for row in rows if row["event"] == "validation"]
    expected_logs = list(range(config["log_freq"], stop + 1, config["log_freq"]))
    expected_eval = list(range(config["eval_steps"], stop + 1, config["eval_steps"]))
    if [row["optimizer_updates"] for row in train] != expected_logs:
        raise ValueError("Missing or repeated logged optimizer updates")
    if [row["optimizer_updates"] for row in validations] != expected_eval:
        raise ValueError("Missing or repeated validation updates")
    for row in rows:
        if not all(
            isinstance(value, (int, float)) and math.isfinite(value) for value in row["metrics"].values()
        ):
            raise ValueError("Non-finite or non-numeric training/validation metrics")
    groups = json.loads((run / "logs/trainable_parameters.json").read_text())
    if groups["backbone"]["trainable"] != 0 or groups["action_head"]["trainable"] != 1_620_515_968:
        raise ValueError("The verified partial-finetuning parameter groups changed")
    events = EventAccumulator(str(run / "logs/tensorboard"), size_guidance={"scalars": 0}).Reload()
    for tag in ("train/loss", "train/gpu_mem_gb", "train/grad_norm"):
        if [value.step for value in events.Scalars(tag)] != expected_logs:
            raise ValueError(f"TensorBoard does not match JSONL: {tag}")
    if [value.step for value in events.Scalars("validation/loss")] != expected_eval:
        raise ValueError("TensorBoard validation points are missing")
    saved_steps = [1000, *range(config["save_freq"], stop + 1, config["save_freq"])]
    saved_steps = sorted({step for step in saved_steps if step <= stop})
    states = [checkpoint_state(run, step, horizon, batch) for step in saved_steps]
    losses = {row["optimizer_updates"]: row["metrics"]["loss"] for row in validations}
    best_step = min(saved_steps, key=lambda step: losses[step])
    report = {
        "status": "passed" if stop == horizon else "phase1_passed",
        "optimizer_updates": stop,
        "training_examples_consumed": stop * batch,
        "nominal_frame_passes": stop * batch / 32910,
        "checkpoints": states,
        "validation_losses": losses,
        "validation_scope": {"episodes": 11, "frames": 4242, "sample_cap": None},
        "best_saved_checkpoint": str(run / "checkpoints" / f"{best_step:06d}"),
        "best_saved_checkpoint_validation_loss": losses[best_step],
        "best_selection_scope": "Lowest held-out loss among the saved checkpoints; test episodes untouched",
        "peak_pytorch_memory_gb": max(row["metrics"]["gpu_mem_gb"] for row in train),
        "parameter_groups": groups,
        "event_files": sorted(path.name for path in (run / "logs/tensorboard").glob("events.out.tfevents.*")),
    }
    if stop == 1000:
        # log_freq=10 reports ten-microbatch averages. Exclude the first window,
        # which includes model/decoder startup, before projecting total runtime.
        steady = [row["metrics"]["step_s"] for row in train[1:]]
        median = statistics.median(steady)
        report["timing"] = {
            "median_seconds_per_update": median,
            "remaining_compute_hours": (horizon - stop) * median / 3600,
            "excludes": "remaining validation, checkpoint writes, process restart and queue time",
        }
        if "timestamp_utc" in validations[-1] and "timestamp_utc" in train[-1]:
            validation_seconds = (
                datetime.fromisoformat(validations[-1]["timestamp_utc"])
                - datetime.fromisoformat(train[-1]["timestamp_utc"])
            ).total_seconds()
            remaining_evaluations = (horizon - stop) // config["eval_steps"]
            report["timing"].update(
                measured_validation_seconds=validation_seconds,
                remaining_validation_hours=remaining_evaluations * validation_seconds / 3600,
            )
    else:
        log = (job / "logs/phase2-resume.log").read_text()
        if "Resuming data order at epoch 0, sample 4000" not in log:
            raise ValueError("No evidence of resuming the sampler after update 1000")
        if len(report["event_files"]) < 2:
            raise ValueError("Expected separate event files before and after process restart")
        report["weight_comparison"] = compare_weights(run)
    return report


def compare_weights(run: Path) -> dict:
    from safetensors import safe_open

    first = run / "checkpoints/001000/pretrained_model/model.safetensors"
    final = run / "checkpoints/030000/pretrained_model/model.safetensors"
    with safe_open(first, framework="np") as start, safe_open(final, framework="np") as end:
        keys = list(start.keys())
        head = next(key for key in keys if "action_head" in key and key.endswith("vlln.weight"))
        frozen = next(key for key in keys if "backbone" in key and "norm" in key and key.endswith("weight"))
        if np.array_equal(start.get_tensor(head), end.get_tensor(head)):
            raise ValueError("The sampled action weight did not change after resume")
        if not np.array_equal(start.get_tensor(frozen), end.get_tensor(frozen)):
            raise ValueError("The sampled frozen backbone weight changed")
    return {"updated_tensor": head, "unchanged_backbone_tensor": frozen}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--expected-stop", type=int, default=30_000)
    args = parser.parse_args()
    result = verify(args.job_root, args.expected_stop)
    name = "verification.json" if args.expected_stop == 30_000 else "phase1-verification.json"
    (args.job_root / name).write_text(json.dumps(result, indent=2) + "\n")
    if args.expected_stop == 30_000:
        (args.job_root / "best_checkpoint.json").write_text(
            json.dumps({key: value for key, value in result.items() if key.startswith("best_")}, indent=2)
            + "\n"
        )
    print(json.dumps(result, indent=2))
