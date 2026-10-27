"""Check offline H200 prerequisites and materialize an absolute-path job config."""

import argparse
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path

from prepare_splits import DATASET_REVISION, MODEL_REVISIONS, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--profile", choices=("h200-smoke100", "4090-environment"), default="h200-smoke100")
    args = parser.parse_args()
    work = args.work_root.resolve()
    job = args.job_root.resolve()
    if not str(work).startswith("/inspire/") or not job.is_relative_to(work):
        raise ValueError("The work and job directories must be on the mounted /inspire shared disk.")
    assets = json.loads((work / "assets.json").read_text())
    if Path(assets["hf_home"]).resolve() != Path(os.environ["HF_HOME"]).resolve():
        raise ValueError("HF_HOME differs from the prepared offline cache.")
    if assets["dataset_revision"] != DATASET_REVISION:
        raise ValueError("Unexpected dataset revision.")
    source = Path(assets["dataset_root"])
    manifest_path = Path(__file__).parent / "splits/split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for relative, expected in manifest["source"]["files_sha256"].items():
        if sha256_file(source / relative) != expected:
            raise ValueError(f"Source checksum mismatch: {relative}")
    # Require every referenced video file before spending GPU time loading the model.
    import pyarrow.parquet as pq

    episodes = pq.read_table(source / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    cameras = [
        key
        for key in json.loads((source / "meta/info.json").read_text())["features"]
        if key.startswith("observation.images.")
    ]
    for row in episodes:
        for camera in cameras:
            relative = f"videos/{camera}/chunk-{row[f'videos/{camera}/chunk_index']:03d}/file-{row[f'videos/{camera}/file_index']:03d}.mp4"
            if not (source / relative).is_file():
                raise FileNotFoundError(relative)
    for repo_id, revision in MODEL_REVISIONS.items():
        model = assets["models"][repo_id]
        if model["revision"] != revision or not (Path(model["path"]) / "config.json").is_file():
            raise ValueError(f"Missing or wrong pinned model: {repo_id}")
        ref = work / "hf/hub" / ("models--" + repo_id.replace("/", "--")) / "refs/main"
        if ref.read_text() != revision:
            raise ValueError(f"Offline model ref changed: {repo_id}")
    import torch

    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.episode_splits import load_episode_splits

    load_episode_splits(manifest_path, "fegkkd/bag_v3", DATASET_REVISION, 110)
    expected_gpu = "4090" if args.profile == "4090-environment" else "H200"
    if torch.cuda.device_count() != 1 or expected_gpu not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"This validation job requires exactly one visible {expected_gpu}.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is unavailable.")
    job.mkdir(parents=True, exist_ok=True)
    config = json.loads((Path(__file__).parent / "train_config.json").read_text())
    config["dataset"].update(root=str(source), split_manifest=str(manifest_path.resolve()))
    config["policy"]["base_model_path"] = assets["models"]["nvidia/GR00T-N1.7-3B"]["path"]
    config["output_dir"] = str(job / "train")
    if args.profile == "4090-environment":
        config.update(stop_after_steps=3, save_freq=3, eval_steps=0, job_name="tianji_wuji_4090_environment")
    target = job / "train_config.json"
    target.write_text(json.dumps(config, indent=2) + "\n")
    # Parse the actual typed schema before allocating the model.
    TrainPipelineConfig.from_pretrained(target.parent)
    versions = {
        name: importlib.metadata.version(name)
        for name in (
            "lerobot",
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
    versions.update(gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda, profile=args.profile)
    (job / "environment.json").write_text(json.dumps(versions, indent=2) + "\n")
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    (job / "code_revision.txt").write_text(result.stdout)
    (job / "working_tree.patch").write_bytes(
        subprocess.run(["git", "diff", "HEAD"], capture_output=True, check=True).stdout
    )
    code_files = [*Path("src/lerobot").rglob("*.py"), Path("pyproject.toml"), Path("uv.lock")]
    code_files.extend(
        path for path in Path(__file__).parent.rglob("*") if path.suffix in (".py", ".json", ".sh", ".md")
    )
    code_manifest = {str(path.resolve().relative_to(Path.cwd())): sha256_file(path) for path in code_files}
    (job / "code_manifest.json").write_text(json.dumps(code_manifest, sort_keys=True, indent=2) + "\n")
    (job / "split_manifest.json").write_bytes(manifest_path.read_bytes())
    (job / "train_stats.json").write_bytes((manifest_path.parent / "train_stats.json").read_bytes())
    (job / "assets.json").write_text(json.dumps(assets, indent=2) + "\n")
    print(json.dumps({"status": "preflight_passed", "gpu": versions["gpu"], "config": str(target)}))


if __name__ == "__main__":
    main()
