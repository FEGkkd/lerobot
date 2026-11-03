"""Materialize and verify the scoped T46 single-GPU cluster runs."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

EXCLUDED = [8105, 8189, 8191, 8218]
MODELS = {
    "nvidia/GR00T-N1.7-3B": "2fc962b973bccdd5d8ce4f67cc63b264d6886495",
    "nvidia/Cosmos-Reason2-2B": "9ce19a195e423419c349abfc86fd07178b230561",
}
TOTAL_STEPS = 1_000_000


def checkpoint_directory(run: Path, step: int) -> Path:
    # Match LeRobot get_step_checkpoint_dir: the million-step horizon uses seven digits.
    return run / "checkpoints" / f"{step:0{max(6, len(str(TOTAL_STEPS)))}d}"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight(job: Path, data: Path, assets_path: Path, profile: str) -> None:
    import torch

    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.episode_splits import load_episode_splits
    from lerobot.policies import GrootConfig

    expected_gpu = "4090" if profile == "4090" else "H200"
    if shutil.disk_usage(job).free < 160 * 1024**3:
        raise RuntimeError("Less than 160 GiB free for separate checkpoints and logs")
    if torch.cuda.device_count() != 1 or expected_gpu not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"Expected exactly one visible {expected_gpu}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is unavailable")
    report = read(data / "validation_report.json")
    if not all(
        report[key]
        for key in (
            "numeric_equal",
            "train_statistics_match_raw",
            "video_full_decode",
            "groot_absolute_action_roundtrip",
        )
    ):
        raise ValueError("Dataset validation is incomplete")
    provenance = read(data / "meta/source_provenance.json")
    if sorted(provenance["excluded_training_episodes"]) != EXCLUDED:
        raise ValueError("The approved four-episode quarantine is missing")
    splits, _ = load_episode_splits(data / "meta/split_manifest.json", "handson/t46_store_charger", None, 296)
    if [len(splits[key]) for key in ("train", "validation", "test")] != [238, 30, 28]:
        raise ValueError("Expected 238 training, 30 validation and 28 held-out test episodes")
    assets = read(assets_path)
    if Path(os.environ["HF_HOME"]).resolve() != Path(assets["hf_home"]).resolve():
        raise ValueError("HF_HOME must use the prepared model cache")
    for repo, revision in MODELS.items():
        model = assets["models"][repo]
        if model["revision"] != revision or not (Path(model["path"]) / "config.json").is_file():
            raise ValueError(f"Missing pinned model: {repo}")
        ref = Path(assets["hf_home"]) / "hub" / ("models--" + repo.replace("/", "--")) / "refs/main"
        if ref.read_text().strip() != revision:
            raise ValueError(f"Offline model reference differs from the pinned revision: {repo}")
    config = read(Path(__file__).with_name("train_config.json"))
    config["dataset"].update(root=str(data), split_manifest=str(data / "meta/split_manifest.json"))
    config["policy"]["base_model_path"] = assets["models"]["nvidia/GR00T-N1.7-3B"]["path"]
    config.update(output_dir=str(job / "train"), job_name=f"handson_t46_{profile}")
    if profile == "4090":
        config.update(stop_after_steps=3, save_freq=3, log_freq=1, eval_steps=1, max_eval_samples=32)
    else:
        config["stop_after_steps"] = 500
    write(job / "train_config.json", config)
    typed = TrainPipelineConfig.from_pretrained(job)
    if not isinstance(typed.policy, GrootConfig):
        raise TypeError("Expected typed GR00T configuration")
    versions = {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "transformers",
            "accelerate",
            "peft",
            "tensorboard",
            "numpy",
            "pyarrow",
            "av",
            "safetensors",
        )
    }
    versions.update(gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda, profile=profile)
    write(job / "environment.json", versions)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    patch = subprocess.check_output(["git", "diff", "HEAD"])
    (job / "working_tree.patch").write_bytes(patch)
    write(job / "code_revision.json", {"head": revision, "patch_sha256": hashlib.sha256(patch).hexdigest()})
    write(
        job / "data_provenance.json",
        {
            "revision": provenance["revision"],
            "excluded_training_episodes": EXCLUDED,
            "train_episodes": 238,
            "validation_episodes": 30,
            "test_episodes": 28,
            "split_sha256": digest(data / "meta/split_manifest.json"),
            "train_stats_sha256": digest(data / "meta/train_stats.json"),
            "validation_report_sha256": digest(data / "validation_report.json"),
        },
    )
    if profile == "h200":
        validation_job = Path(os.environ["HANDSON_VALIDATION_JOB"])
        prior = read(validation_job / "verification.json")
        if prior.get("status") != "passed" or prior.get("profile") != "4090":
            raise ValueError("H200 requires a passed 4090 validation report")
        for name in ("code_revision.json", "data_provenance.json"):
            if read(job / name) != read(validation_job / name):
                raise ValueError(f"H200 differs from the GPU-tested code/data: {name}")
    print(json.dumps({"status": "preflight_passed", "gpu": versions["gpu"], "profile": profile}))


def checkpoint(run: Path, step: int) -> dict:
    from safetensors import safe_open

    root = checkpoint_directory(run, step)
    state = root / "training_state"
    metadata = read(state / "training_step.json")
    if (metadata["step"], metadata["grad_accum_steps"], metadata["dp_world_size"]) != (step, 1, 1):
        raise ValueError("Checkpoint step or topology mismatch")
    scheduler = read(state / "scheduler_state.json")
    if scheduler["last_epoch"] != step:
        raise ValueError("Scheduler counter mismatch")
    for name in ("optimizer_param_groups.json", "rng_state.safetensors"):
        if not (state / name).is_file():
            raise FileNotFoundError(state / name)
    with safe_open(state / "optimizer_state.safetensors", framework="np") as tensors:
        keys = list(tensors.keys())
        counters = [float(tensors.get_tensor(key).item()) for key in keys if key.endswith("/step")]
    if not counters or min(counters) != step or max(counters) != step:
        raise ValueError("Adam counters are not all continuous")
    cfg = read(root / "pretrained_model/train_config.json")
    if (
        cfg["steps"],
        cfg["policy"]["max_steps"],
        cfg["batch_size"],
        cfg["scheduler"]["num_warmup_steps"],
    ) != (1000000, 1000000, 4, 100):
        raise ValueError("Formal scheduler horizon or batch changed during the test")
    return {"step": step, "adam_min": min(counters), "adam_max": max(counters), "scheduler": step}


def verify(job: Path, profile: str) -> None:
    import numpy as np
    from safetensors import safe_open

    run = job / "train"
    records = [json.loads(line) for line in (run / "logs/metrics.jsonl").read_text().splitlines()]
    train = [row for row in records if row["event"] == "train"]
    validations = [row["optimizer_updates"] for row in records if row["event"] == "validation"]
    final_step = 4 if profile == "4090" else read(job / "progress.json")["completed_updates"]
    expected = list(range(1, 5)) if profile == "4090" else list(range(10, final_step + 1, 10))
    expected_val = list(range(1, 5)) if profile == "4090" else list(range(100, final_step + 1, 100))
    if [row["optimizer_updates"] for row in train] != expected or validations != expected_val:
        raise ValueError("Missing or repeated training/validation metrics")
    for row in records:
        for value in row["metrics"].values():
            if isinstance(value, str) or not math.isfinite(value):
                raise ValueError("Non-finite or nonnumeric metric")
    groups = read(run / "logs/trainable_parameters.json")
    if groups["backbone"]["trainable"] or groups["action_head"]["trainable"] <= 0:
        raise ValueError("Partial fine-tuning freeze mismatch")
    if not list((run / "logs/tensorboard").glob("events.out.tfevents.*")):
        raise FileNotFoundError("TensorBoard event files")
    steps = (3, 4) if profile == "4090" else tuple(range(500, final_step + 1, 500))
    checkpoints = [checkpoint(run, step) for step in steps]
    paths = [checkpoint_directory(run, step) / "pretrained_model/model.safetensors" for step in steps]
    with safe_open(paths[0], framework="np") as first, safe_open(paths[-1], framework="np") as last:
        keys = list(first.keys())
        head = next(key for key in keys if "action_head" in key and key.endswith("vlln.weight"))
        frozen = next(key for key in keys if "backbone" in key and "norm" in key and key.endswith("weight"))
        if len(paths) > 1 and np.array_equal(first.get_tensor(head), last.get_tensor(head)):
            raise ValueError("Trainable head sample did not change")
        if not np.array_equal(first.get_tensor(frozen), last.get_tensor(frozen)):
            raise ValueError("Frozen backbone sample changed")
    result = {
        "status": "passed",
        "profile": profile,
        "optimizer_updates": expected[-1],
        "checkpoints": checkpoints,
        "validation_updates": validations,
        "validation_sample_cap": 32 if profile == "4090" else 0,
        "validation_episodes": 30,
        "held_out_test_episodes": 28,
        "test_used_for_model_selection": False,
        "parameter_groups": groups,
        "updated_tensor": head,
        "unchanged_backbone_tensor": frozen,
        "closed_loop_success_measured": False,
    }
    if profile == "4090":
        inference = read(job / "inference.json")
        if inference.get("status") != "passed" or inference["shape"] != [1, 16, 52]:
            raise ValueError("Saved-policy inference did not pass")
        result["saved_policy_inference"] = inference
    write(job / "verification.json", result)
    print(json.dumps(result, indent=2))


def inference(job: Path) -> None:
    import torch
    from torch.utils.data import DataLoader

    from lerobot.datasets import LeRobotDataset
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.groot.modeling_groot import GrootPolicy

    saved = checkpoint_directory(job / "train", 4) / "pretrained_model"
    cfg = read(saved / "train_config.json")
    splits = read(Path(cfg["dataset"]["split_manifest"]))["splits"]
    dataset = LeRobotDataset(
        "handson/t46_store_charger",
        root=Path(cfg["dataset"]["root"]),
        episodes=splits["validation"],
        video_backend="pyav",
        return_uint8=True,
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
    policy = GrootPolicy.from_pretrained(saved)
    policy.to("cuda").eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=saved,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    with torch.inference_mode():
        action = post(policy.predict_action_chunk(pre(batch)))
    if list(action.shape) != [1, 16, 52] or not torch.isfinite(action).all():
        raise ValueError("Restored-policy decoded absolute actions are invalid")
    write(
        job / "inference.json",
        {
            "status": "passed",
            "shape": list(action.shape),
            "device": str(action.device),
            "absolute_controller_command": True,
            "source_split": "validation",
            "min": float(action.min()),
            "max": float(action.max()),
            "closed_loop_success_measured": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "verify", "inference"))
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--profile", choices=("4090", "h200"), required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--assets", type=Path)
    args = parser.parse_args()
    if args.command == "preflight":
        preflight(args.job_root.resolve(), args.data_root.resolve(), args.assets.resolve(), args.profile)
    elif args.command == "verify":
        verify(args.job_root.resolve(), args.profile)
    else:
        inference(args.job_root.resolve())


if __name__ == "__main__":
    main()
