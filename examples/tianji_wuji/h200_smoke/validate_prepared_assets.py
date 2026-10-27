"""Validate the prepared dataset and model cache with networking disabled."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from prepare_splits import MODEL_REVISIONS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--wait-minutes", type=int, default=0)
    args = parser.parse_args()
    work = args.work_root.resolve()
    deadline = time.monotonic() + args.wait_minutes * 60
    while not (work / "assets.json").is_file():
        if time.monotonic() >= deadline:
            raise RuntimeError("Verified shared assets are not ready")
        time.sleep(10)
    os.environ.update(HF_HOME=str(work / "hf"), HF_HUB_CACHE=str(work / "hf/hub"), HF_HUB_OFFLINE="1")

    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoConfig

    from lerobot.policies.groot.processor_groot import _build_n1_7_processor

    assets = json.loads((work / "assets.json").read_text())
    model_checks = {}
    for repo, revision in MODEL_REVISIONS.items():
        snapshot = Path(assets["models"][repo]["path"]).resolve()
        cached = Path(
            hf_hub_download(repo, "config.json", revision=revision, local_files_only=True)
        ).resolve()
        cached_main = Path(
            hf_hub_download(repo, "config.json", revision="main", local_files_only=True)
        ).resolve()
        if cached != (snapshot / "config.json").resolve() or cached_main != cached:
            raise ValueError(f"Offline cache resolved a different snapshot: {repo}")
        keys = set()
        files = list(snapshot.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"Model weight files missing: {repo}")
        for file in files:
            with safe_open(file, framework="pt", device="cpu") as weights:
                keys.update(weights.keys())
        index = snapshot / "model.safetensors.index.json"
        if not keys or (index.is_file() and keys != set(json.loads(index.read_text())["weight_map"])):
            raise ValueError(f"Model weight header/index mismatch: {repo}")
        model_checks[repo] = {"revision": revision, "weight_files": len(files), "tensor_keys": len(keys)}
    config = AutoConfig.from_pretrained("nvidia/Cosmos-Reason2-2B", local_files_only=True)
    if config.model_type != "qwen3_vl":
        raise ValueError("Unexpected Cosmos backbone configuration")
    processor = _build_n1_7_processor()
    if processor.tokenizer.padding_side != "left":
        raise ValueError("Unexpected tokenizer padding")
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("prepare_splits.py")),
            "--root",
            assets["dataset_root"],
            "--verify-only",
        ],
        check=True,
    )
    report = {
        "status": "offline_assets_validation_passed",
        "models": model_checks,
        "cosmos_processor_offline": True,
        "dataset_splits_and_statistics_verified": True,
        "gpu_execution_tested": False,
    }
    (work / "preparation/offline-assets-validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
