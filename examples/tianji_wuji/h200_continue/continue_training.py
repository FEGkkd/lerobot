"""Bounded H200 continuation: preserve model/Adam/RNG, restart only the LR schedule."""

import argparse
import json
import math
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lerobot.configs.train import TrainPipelineConfig

START = 30_000
HORIZON = 300_000
PHASE = 5_000
LR = 1e-5
WARMUP = 500
TARGET = 0.001
SOURCE_JOB = "tianji-h200-formal-v1-20261028-143520"


def load_resume_config(config_path: Path) -> "TrainPipelineConfig":
    """Register the policy in this process before decoding a checkpoint recipe."""
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies import GrootConfig

    typed = TrainPipelineConfig.from_pretrained(config_path.parent)
    if not isinstance(typed.policy, GrootConfig):
        raise TypeError("Expected the registered GR00T policy configuration")
    if typed.scheduler.type != "continuation_constant" or typed.batch_size != 4:
        raise ValueError("Typed continuation recipe differs")
    return typed


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def moving_loss(train: list[dict]) -> float:
    """Mean of100 consecutive ten-update logged averages, not the minimum sample."""
    window = train[-100:]
    if len(window) != 100:
        raise ValueError("Insufficient moving-loss window")
    values = [x["metrics"]["loss"] for x in window]
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Non-finite moving-loss window")
    return statistics.mean(values)


def stage_resume(source_job: Path, job: Path) -> Path:
    """Create an independent resume view; never edit the source checkpoint or its logs."""
    result = read(source_job / "verification.json")
    if result["status"] != "passed" or result["optimizer_updates"] != START:
        raise ValueError("Source must be the verified 30000-update formal run")
    source = source_job / "train/checkpoints/030000"
    state = read(source / "training_state/training_step.json")
    if (state["step"], state["batch_size"], state["grad_accum_steps"], state["dp_world_size"]) != (
        START,
        4,
        1,
        1,
    ):
        raise ValueError("Unexpected source update count, batch or topology")
    config = read(source / "pretrained_model/train_config.json")
    if config["optimizer"]["type"] != "adamw" or config["batch_size"] != 4:
        raise ValueError("Only the verified AdamW/batch4 recipe is supported")
    for relative in (
        "pretrained_model/model.safetensors",
        "training_state/optimizer_state.safetensors",
        "training_state/rng_state.safetensors",
        "training_state/scheduler_state.json",
        "training_state/optimizer_param_groups.json",
    ):
        if not (source / relative).is_file():
            raise FileNotFoundError(source / relative)
    target = job / "resume/checkpoints/030000"
    target.mkdir(parents=True, exist_ok=False)
    for item in source.rglob("*"):
        dest = target / item.relative_to(source)
        if item.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            if item.suffix == ".json":
                shutil.copy2(item, dest)
            else:
                dest.symlink_to(item.resolve())
    config.update(
        resume=True,
        output_dir=str(job / "train"),
        job_name="tianji_wuji_h200_continue_loss001",
        steps=HORIZON,
        stop_after_steps=START + PHASE,
        batch_size=4,
        save_freq=PHASE,
        eval_steps=1000,
        max_eval_samples=0,
        log_freq=10,
        local_metrics=True,
        use_policy_training_preset=False,
        scheduler={"type": "continuation_constant", "start_step": START, "num_warmup_steps": WARMUP},
    )
    config["optimizer"]["lr"] = LR
    config["policy"].update(max_steps=HORIZON, optimizer_lr=LR)
    config["accelerator"]["gradient_accumulation"]["steps"] = 1
    write(target / "pretrained_model/train_config.json", config)
    groups_path = target / "training_state/optimizer_param_groups.json"
    groups = read(groups_path)
    initial_lr = LR / WARMUP
    for group in groups:
        group.update(lr=initial_lr, initial_lr=LR)
    write(groups_path, groups)
    scheduler_path = target / "training_state/scheduler_state.json"
    scheduler = read(scheduler_path)
    if scheduler["last_epoch"] != START:
        raise ValueError("Source scheduler is not at update30000")
    scheduler.update(
        base_lrs=[LR] * len(groups),
        _last_lr=[initial_lr] * len(groups),
        last_epoch=START,
        _step_count=START + 1,
        lr_lambdas=[None] * len(groups),
    )
    write(scheduler_path, scheduler)
    write(job / "train_config.json", config)
    write(
        job / "continuation.json",
        {
            "source_job": str(source_job),
            "source_checkpoint": str(source),
            "start_step": START,
            "horizon": HORIZON,
            "lr": LR,
            "warmup_updates": WARMUP,
            "schedule": "warmup from global30000 then constant1e-5; Adam moments/RNG/counters retained",
            "threshold": TARGET,
            "window_updates": 1000,
            "check_every_updates": PHASE,
            "max_job_hours": 12,
            "automatic_platform_resubmission": False,
            "source_model_and_adam": "read-only references; original files and logs untouched",
        },
    )
    return target / "pretrained_model/train_config.json"


def verify_phase(job: Path, stop: int) -> dict:
    from safetensors import safe_open
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    run = job / "train"
    rows = [json.loads(x) for x in (run / "logs/metrics.jsonl").read_text().splitlines() if x.strip()]
    train = [x for x in rows if x["event"] == "train"]
    val = [x for x in rows if x["event"] == "validation"]
    expected = list(range(START + 10, stop + 1, 10))
    if [x["optimizer_updates"] for x in train] != expected:
        raise ValueError("Missing/repeated continuation updates")
    if [x["optimizer_updates"] for x in val] != list(range(START + 1000, stop + 1, 1000)):
        raise ValueError("Missing/repeated validation updates")
    for row in rows:
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in row["metrics"].values()):
            raise ValueError("Non-finite metrics")
    if any(x["metrics"]["samples"] != 4 * x["optimizer_updates"] for x in train):
        raise ValueError("Batch/sample counter changed")
    groups = read(run / "logs/trainable_parameters.json")
    if groups["backbone"]["trainable"] != 0 or groups["action_head"]["trainable"] != 1620515968:
        raise ValueError("Partial-tuning parameter groups changed")
    events = EventAccumulator(str(run / "logs/tensorboard"), size_guidance={"scalars": 0}).Reload()
    for tag in ("train/loss", "train/gpu_mem_gb", "train/lr"):
        if [x.step for x in events.Scalars(tag)] != expected:
            raise ValueError(f"TensorBoard history differs: {tag}")
    if [x.step for x in events.Scalars("validation/loss")] != [x["optimizer_updates"] for x in val]:
        raise ValueError("TensorBoard validation history differs")
    cp = run / "checkpoints" / f"{stop:06d}"
    state = read(cp / "training_state/training_step.json")
    scheduler = read(cp / "training_state/scheduler_state.json")
    if (state["step"], state["batch_size"], state["grad_accum_steps"], scheduler["last_epoch"]) != (
        stop,
        4,
        1,
        stop,
    ):
        raise ValueError("Checkpoint update/scheduler count differs")
    with safe_open(cp / "training_state/optimizer_state.safetensors", framework="np") as tensors:
        adam = [float(tensors.get_tensor(k).item()) for k in list(tensors.keys()) if k.endswith("/step")]
    if not adam or max(adam) != stop:
        raise ValueError("Adam update count was not retained")
    saved = read(cp / "pretrained_model/train_config.json")
    if saved["steps"] != HORIZON or saved["scheduler"]["type"] != "continuation_constant":
        raise ValueError("The continuation schedule changed")
    if any(abs(x - LR) > 1e-12 for x in scheduler["_last_lr"]):
        raise ValueError("Continuation LR differs from the post-warmup constant")
    if stop == START + PHASE:
        log = (job / "logs" / f"phase-{stop:06d}.log").read_text()
        if "Resuming data order at epoch" not in log:
            raise ValueError("Missing sampler-resume evidence")
    # Each logged loss averages ten updates, so 100 equal-weight entries cover1000 updates.
    mean_loss = moving_loss(train)
    report = {
        "status": "phase_passed",
        "start_step": START,
        "optimizer_updates": stop,
        "additional_updates": stop - START,
        "optimizer_step": max(adam),
        "scheduler_step": scheduler["last_epoch"],
        "window_updates": 1000,
        "train_loss_window_mean": mean_loss,
        "target": TARGET,
        "target_reached": mean_loss <= TARGET,
        "validation_loss": val[-1]["metrics"]["loss"],
        "validation_losses": {str(x["optimizer_updates"]): x["metrics"]["loss"] for x in val},
        "peak_pytorch_memory_gb": max(x["metrics"]["gpu_mem_gb"] for x in train),
        "checkpoint": str(cp),
        "event_files": sorted(x.name for x in (run / "logs/tensorboard").glob("events.*")),
        "parameter_groups": groups,
    }
    write(job / "progress.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--job-root", type=Path, required=True)
    args = parser.parse_args()
    work, job = args.work_root.resolve(), args.job_root.resolve()
    if not job.is_relative_to(work / "jobs") or not str(work).startswith("/inspire/"):
        raise ValueError("Continuation must write inside the personal shared jobs directory")
    if job.exists():
        raise FileExistsError("Job already exists; do not start a duplicate process")
    if shutil.disk_usage(work).free < 100 * 1024**3:
        raise RuntimeError("Require at least100GiB free before continuation")
    (job / "logs").mkdir(parents=True)
    source = work / "jobs" / SOURCE_JOB
    resume_config = stage_resume(source, job)
    subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "python",
            "examples/tianji_wuji/h200_smoke/preflight.py",
            "--work-root",
            str(work),
            "--job-root",
            str(job),
            "--profile",
            "h200-formal",
            "--config",
            str(job / "train_config.json"),
        ],
        check=True,
    )
    # Parse the staged typed schema too; the preflight config itself is a provenance snapshot.
    load_resume_config(resume_config)
    started, previous_phase_seconds = time.monotonic(), 2400.0
    report = None
    reason = "step_cap"
    stop_file = job / "STOP_AFTER_PHASE"
    for stop in range(START + PHASE, HORIZON + 1, PHASE):
        if shutil.disk_usage(work).free < 60 * 1024**3:
            reason = "storage_limit"
            break
        if time.monotonic() - started + max(2400, previous_phase_seconds * 1.25) + 600 > 11.5 * 3600:
            reason = "runtime_budget"
            break
        phase_started = time.monotonic()
        command = [
            "uv",
            "run",
            "--no-sync",
            "lerobot-train",
            f"--config_path={resume_config}",
            "--resume=true",
            f"--stop_after_steps={stop}",
        ]
        log_path = job / "logs" / f"phase-{stop:06d}.log"
        # tee preserves full platform and persistent logs; parent shell has pipefail too.
        with log_path.open("w") as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if process.wait():
                raise RuntimeError(f"Training phase failed; inspect {log_path}")
        report = verify_phase(job, stop)
        print(json.dumps(report), flush=True)
        previous_phase_seconds = time.monotonic() - phase_started
        resume_config = Path(report["checkpoint"]) / "pretrained_model/train_config.json"
        if report["target_reached"]:
            reason = "target_reached"
            break
        if stop_file.exists():
            reason = "user_or_monitor_stop"
            break
    if report is None:
        raise RuntimeError("No continuation phase completed; inspect runtime and storage")
    report.update(status="passed", stop_reason=reason, elapsed_hours=(time.monotonic() - started) / 3600)
    losses = report["validation_losses"]
    saved_steps = list(range(START + PHASE, report["optimizer_updates"] + 1, PHASE))
    best_step = min(saved_steps, key=lambda s: losses[str(s)])
    write(
        job / "best_checkpoint.json",
        {
            "best_saved_checkpoint": str(job / "train/checkpoints" / f"{best_step:06d}"),
            "best_saved_checkpoint_validation_loss": losses[str(best_step)],
            "scope": "saved continuation checkpoints only; original020000 remains available separately",
        },
    )
    write(job / "verification.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
