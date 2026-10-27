"""Download pinned assets on an Internet-enabled Linux preparation machine."""

import argparse
import json
from pathlib import Path

from prepare_splits import DATASET_REVISION, MODEL_REVISIONS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.work_root.resolve()
    from huggingface_hub import HfApi, snapshot_download

    cache = root / "hf" / "hub"
    data_root = root / "datasets/bag_v3"
    snapshot_download("fegkkd/bag_v3", repo_type="dataset", revision=DATASET_REVISION, local_dir=data_root)
    models = {}
    for repo_id, revision in MODEL_REVISIONS.items():
        # Resolve main, check its pinned SHA, then download main to populate the offline ref
        # expected by the native GR00T backbone loader. Check again after download for races.
        if HfApi().model_info(repo_id, revision="main").sha != revision:
            raise RuntimeError(f"{repo_id} main changed; review model pins before downloading.")
        snapshot = Path(snapshot_download(repo_id, revision="main", cache_dir=cache))
        if snapshot.name != revision:
            raise RuntimeError(f"Model revision changed during preparation: {repo_id}")
        models[repo_id] = {"revision": revision, "path": str(snapshot)}
    pins = {
        "dataset_root": str(data_root),
        "dataset_revision": DATASET_REVISION,
        "hf_home": str(root / "hf"),
        "models": models,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "assets.json").write_text(json.dumps(pins, indent=2) + "\n")
    print("Pinned dataset and both models prepared; assets.json contains their local paths.")


if __name__ == "__main__":
    main()
