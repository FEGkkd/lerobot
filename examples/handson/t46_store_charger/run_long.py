"""Run one bounded H200 job in 500-update resumable phases until manually stopped."""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from examples.handson.t46_store_charger.cluster_check import checkpoint, checkpoint_directory, read, write


def stop_reason(job: Path, remaining: float, phase_seconds: float, free_bytes: int) -> str | None:
    if (job / "STOP_AFTER_PHASE").exists():
        return "user_stop"
    if remaining < max(1800, phase_seconds * 1.5 + 180):
        return "runtime_budget"
    if free_bytes < 60 * 1024**3:
        return "storage_budget"
    return None


def execute(command: list[str], log: Path, remaining: float) -> int:
    """Stream logs and terminate the entire training process group at the job deadline."""
    with log.open("x") as output:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            bufsize=1,
        )

        def copy_output() -> None:
            for line in process.stdout:
                output.write(line)
                output.flush()
                print(line, end="", flush=True)

        reader = threading.Thread(target=copy_output, daemon=True)
        reader.start()
        try:
            code = process.wait(timeout=max(1, remaining))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = 124
        reader.join(timeout=10)
    return code


def run(job: Path) -> None:
    config = read(job / "train_config.json")
    if config["steps"] != 1000000 or config["max_eval_samples"] != 0:
        raise ValueError("Expected the approved million-update horizon and full validation")
    started = time.monotonic()
    deadline = started + 11.5 * 3600
    completed = 0
    phase_seconds = 0.0
    reason = None
    for target in range(500, config["steps"] + 1, 500):
        reason = stop_reason(job, deadline - time.monotonic(), phase_seconds, shutil.disk_usage(job).free)
        if reason:
            break
        source = (
            job / "train_config.json"
            if completed == 0
            else (checkpoint_directory(job / "train", completed) / "pretrained_model/train_config.json")
        )
        command = [sys.executable, "-m", "lerobot.scripts.lerobot_train", f"--config_path={source}"]
        if completed:
            command.append("--resume=true")
        command.append(f"--stop_after_steps={target}")
        phase_started = time.monotonic()
        code = execute(command, job / f"logs/phase-{target:06d}.log", deadline - phase_started)
        phase_seconds = time.monotonic() - phase_started
        if code:
            reason = "runtime_budget_interrupted_phase" if code == 124 else "training_failure"
            break
        state = checkpoint(job / "train", target)
        completed = target
        rows = [json.loads(line) for line in (job / "train/logs/metrics.jsonl").read_text().splitlines()]
        saved_validations = [
            row for row in rows if row["event"] == "validation" and row["optimizer_updates"] % 500 == 0
        ]
        best = min(saved_validations, key=lambda row: row["metrics"]["loss"])
        write(
            job / "best_checkpoint.json",
            {
                "step": best["optimizer_updates"],
                "validation_loss": best["metrics"]["loss"],
                "path": str(checkpoint_directory(job / "train", best["optimizer_updates"])),
                "selected_using": "full validation split only; test split remains unused",
            },
        )
        write(
            job / "progress.json",
            {
                "status": "phase_passed",
                "completed_updates": completed,
                "checkpoint": state,
                "phase_seconds": phase_seconds,
                "elapsed_seconds": time.monotonic() - started,
                "last_validation_loss": saved_validations[-1]["metrics"]["loss"],
                "test_evaluated": False,
            },
        )
    else:
        reason = "step_upper_bound"
    write(
        job / "stop.json",
        {
            "stop_reason": reason,
            "completed_phase_updates": completed,
            "elapsed_seconds": time.monotonic() - started,
            "loss_target_defined": False,
            "automatic_resubmission": False,
            "test_evaluated": False,
        },
    )
    if reason in ("training_failure", "runtime_budget_interrupted_phase") or completed == 0:
        raise RuntimeError(
            f"Stopped with {reason}; retain logs and latest complete checkpoint at {completed}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.job_root.resolve())
