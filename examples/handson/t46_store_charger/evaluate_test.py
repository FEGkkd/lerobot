"""Evaluate the validation-selected checkpoint on held-out sessions after training stops."""

import argparse
import datetime as dt
import math
import signal
import time
from pathlib import Path

from examples.handson.t46_store_charger.cluster_check import checkpoint_directory, read, write


def evaluate(job: Path) -> None:
    import torch
    from torch.utils.data import DataLoader

    from lerobot.datasets import EpisodeAwareSampler, LeRobotDataset
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.groot.modeling_groot import GrootPolicy

    output = job / "test-evaluation.json"
    if output.exists():
        raise FileExistsError("Final test already recorded; do not use repeated test scores to select models")
    acceptance, stopped = read(job / "verification.json"), read(job / "stop.json")
    if acceptance["status"] != "passed" or acceptance["profile"] != "h200":
        raise ValueError("Final test requires an accepted, stopped H200 training run")
    if stopped["stop_reason"] not in ("user_stop", "runtime_budget", "storage_budget", "step_upper_bound"):
        raise ValueError("Resolve incomplete training phases before final testing")
    started_at = dt.datetime.fromisoformat(read(job / "progress.json")["job_started_at_utc"])
    remaining = 12 * 3600 - (dt.datetime.now(dt.UTC) - started_at).total_seconds() - 180
    validation_seconds = read(job / "progress.json")["last_validation_seconds"]
    estimate = max(300, validation_seconds * (5494 / 5869) * 1.5 + 180)
    if remaining < estimate:
        write(
            job / "test-evaluation-status.json",
            {
                "status": "pending_runtime_budget",
                "remaining_seconds": remaining,
                "estimated_seconds": estimate,
                "test_score_computed": False,
            },
        )
        return

    def timed_out(signum, frame):
        raise TimeoutError("Final test reached the existing job's remaining time budget")

    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(max(1, int(remaining)))
    started = time.monotonic()
    try:
        best = read(job / "best_checkpoint.json")
        saved = checkpoint_directory(job / "train", best["step"]) / "pretrained_model"
        cfg = read(saved / "train_config.json")
        splits = read(Path(cfg["dataset"]["split_manifest"]))["splits"]
        if len(splits["test"]) != 28 or set(splits["test"]) & (
            set(splits["train"]) | set(splits["validation"])
        ):
            raise ValueError("Held-out test membership mismatch")
        dataset = LeRobotDataset(
            "handson/t46_store_charger",
            root=Path(cfg["dataset"]["root"]),
            episodes=splits["test"],
            video_backend="pyav",
            return_uint8=True,
            delta_timestamps={"action": [i / 30 for i in range(16)]},
        )
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=15,
            absolute_to_relative_idx=dataset.absolute_to_relative_idx,
        )
        if len(sampler) != 5494:
            raise ValueError("Expected all 5494 complete test action chunks")
        loader = DataLoader(
            dataset,
            batch_size=4,
            sampler=sampler,
            num_workers=4,
            multiprocessing_context="spawn",
            pin_memory=True,
            generator=torch.Generator().manual_seed(42),
        )
        torch.manual_seed(42)
        policy = GrootPolicy.from_pretrained(saved).to("cuda").eval()
        pre, _ = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=saved,
            preprocessor_overrides={"device_processor": {"device": "cuda"}},
        )
        weighted_loss, samples = 0.0, 0
        with torch.inference_mode():
            for batch in loader:
                size = batch["action"].shape[0]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=policy.config.use_bf16):
                    loss, _ = policy(pre(batch))
                value = float(loss)
                if not math.isfinite(value):
                    raise ValueError("Non-finite final test loss")
                weighted_loss += value * size
                samples += size
        if samples != 5494:
            raise ValueError("Final test did not evaluate every held-out complete action chunk")
        write(
            output,
            {
                "status": "passed",
                "checkpoint_step": best["step"],
                "selection_split": "validation",
                "test_episodes": 28,
                "samples": samples,
                "loss": weighted_loss / samples,
                "seed": 42,
                "elapsed_seconds": time.monotonic() - started,
                "normalization": "saved train-only statistics",
                "used_for_model_selection": False,
                "closed_loop_success_measured": False,
            },
        )
    except TimeoutError as error:
        write(
            job / "test-evaluation-status.json",
            {
                "status": "pending_runtime_budget",
                "error": str(error),
                "test_score_computed": False,
            },
        )
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.job_root.resolve())
