"""Exercise real event/checkpoint formats without loading the GPU policy."""

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

TASK = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("formal_verifier", TASK / "verify_formal.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def fixture_job(root: Path, stop=30_000):
    config = json.loads((TASK / "train_config.json").read_text())
    dump(root / "train_config.json", config)
    logs = root / "train/logs"
    logs.mkdir(parents=True)
    rows = []
    segments = [(10, 1000)] if stop == 1000 else [(10, 1000), (1010, 30_000)]
    for first, last in segments:
        writer = EventFileWriter(str(logs / "tensorboard"))
        for step in range(first, last + 1, 10):
            metrics = {"loss": 1.0, "grad_norm": 1.0, "gpu_mem_gb": 55.0, "step_s": 0.6}
            rows.append({"event": "train", "optimizer_updates": step, "metrics": metrics})
            writer.add_event(
                Event(
                    wall_time=time.time(),
                    step=step,
                    summary=Summary(
                        value=[
                            Summary.Value(tag=f"train/{key}", simple_value=value)
                            for key, value in metrics.items()
                        ]
                    ),
                )
            )
            if step % 1000 == 0:
                loss = 0.1 if step == 2000 else 0.5 if step == 5000 else 1.0
                rows.append({"event": "validation", "optimizer_updates": step, "metrics": {"loss": loss}})
                writer.add_event(
                    Event(
                        wall_time=time.time(),
                        step=step,
                        summary=Summary(value=[Summary.Value(tag="validation/loss", simple_value=loss)]),
                    )
                )
        writer.close()
    (logs / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    dump(
        logs / "trainable_parameters.json",
        {
            "backbone": {"trainable": 0, "frozen": 1_523_500_032},
            "action_head": {"trainable": 1_620_515_968, "frozen": 0},
        },
    )
    for step in [1000, *range(5000, stop + 1, 5000)]:
        checkpoint = root / "train/checkpoints" / f"{step:06d}"
        state = checkpoint / "training_state"
        dump(
            state / "training_step.json",
            {"step": step, "grad_accum_steps": 1, "dp_world_size": 1, "batch_size": 4},
        )
        dump(state / "scheduler_state.json", {"last_epoch": step})
        dump(state / "optimizer_param_groups.json", [])
        save_file({"rng": np.array([42])}, state / "rng_state.safetensors")
        save_file({"state/0/step": np.array(float(step))}, state / "optimizer_state.safetensors")
        dump(checkpoint / "pretrained_model/train_config.json", config)
        save_file(
            {
                "_groot_model.action_head.vlln.weight": np.array([float(step)]),
                "_groot_model.backbone.norm.weight": np.array([1.0]),
            },
            checkpoint / "pretrained_model/model.safetensors",
        )
    phase2 = root / "logs/phase2-resume.log"
    phase2.parent.mkdir()
    phase2.write_text("Resuming data order at epoch 0, sample 4000\n")
    return config


def test_full_run_selects_only_an_existing_saved_checkpoint(tmp_path):
    fixture_job(tmp_path)
    report = verifier.verify(tmp_path, 30_000)
    assert report["status"] == "passed"
    assert report["training_examples_consumed"] == 120_000
    assert len(report["checkpoints"]) == 7
    # Update2000 has the lowest loss but is not a saved checkpoint.
    assert report["best_saved_checkpoint"].endswith("005000")
    assert report["best_saved_checkpoint_validation_loss"] == 0.5
    assert report["weight_comparison"]["updated_tensor"].endswith("vlln.weight")


def test_startup_check_keeps_the_full_scheduler_horizon(tmp_path):
    fixture_job(tmp_path, 1000)
    report = verifier.verify(tmp_path, 1000)
    assert report["status"] == "phase1_passed"
    assert report["optimizer_updates"] == 1000
    assert report["timing"]["median_seconds_per_update"] == 0.6
    assert report["timing"]["remaining_compute_hours"] == pytest.approx(29_000 * 0.6 / 3600)


@pytest.mark.parametrize("corruption", ["optimizer", "missing_update", "resume"])
def test_incomplete_or_reinitialized_runs_are_rejected(tmp_path, corruption):
    fixture_job(tmp_path)
    if corruption == "optimizer":
        save_file(
            {"state/0/step": np.array(29_000.0)},
            tmp_path / "train/checkpoints/030000/training_state/optimizer_state.safetensors",
        )
    elif corruption == "missing_update":
        path = tmp_path / "train/logs/metrics.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        path.write_text(
            "\n".join(
                json.dumps(row)
                for row in rows
                if not (row["event"] == "train" and row["optimizer_updates"] == 1010)
            )
        )
    else:
        (tmp_path / "logs/phase2-resume.log").write_text("optimizer initialized from scratch")
    with pytest.raises(ValueError):
        verifier.verify(tmp_path, 30_000)
