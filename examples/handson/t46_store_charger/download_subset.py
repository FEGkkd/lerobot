"""Download only selected HandsOn T46 episodes from a pinned dataset revision."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "Cyan666/HandsOn-Datasets"
REVISION = "7ad38776d91e4e13c02dad4e6ad892e8b8a1d187"
SOURCE_TASK = "bench320_fu006_hold_drawer_store_charger"
DEFAULT_ROOT = Path("/home/ubuntu/lerobot_ws/datasets/handson_t46/raw")
METADATA = (
    "meta/info.json",
    "meta/episodes.jsonl",
    "meta/episodes_stats.jsonl",
    "meta/tasks.jsonl",
    "meta/splits.json",
    "meta/handson_policy_schema.json",
    "source_manifest.jsonl",
    "FILE_MANIFEST.jsonl",
    "code/training_reader.py",
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def select_episodes(
    rows: list[dict], splits: dict, train_count: int | None, validation_count: int | None
) -> tuple[list[dict], dict[str, int]]:
    by_split = {name: set(splits[name]) for name in ("train", "validation")}
    matching = [row for row in rows if row["source_task"] == SOURCE_TASK]
    for row in matching:
        membership = [name for name, indices in by_split.items() if row["episode_index"] in indices]
        if membership != [row["split"]]:
            raise ValueError(
                f"split disagreement for episode {row['episode_index']}: {membership} vs {row['split']}"
            )
    counts = dict(Counter(row["split"] for row in matching))
    counts = {name: counts.get(name, 0) for name in ("train", "validation")}
    selected = []
    for name, limit in (("train", train_count), ("validation", validation_count)):
        candidates = sorted(
            (row for row in matching if row["split"] == name),
            key=lambda row: (row["frames"], row["episode_index"]),
        )
        if limit is not None and len(candidates) < limit:
            raise ValueError(f"Only {len(candidates)} {name} T46 episodes, requested {limit}")
        selected.extend(candidates if limit is None else candidates[:limit])
    return selected, counts


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(root: Path, revision: str, train_count: int | None, validation_count: int | None) -> dict:
    resolved = HfApi().dataset_info(REPO_ID, revision=revision).sha
    if resolved != revision:
        raise ValueError(f"Expected fixed revision {revision}, got {resolved}")
    for filename in METADATA:
        hf_hub_download(REPO_ID, filename=filename, repo_type="dataset", revision=revision, local_dir=root)
    selected, counts = select_episodes(
        read_jsonl(root / "source_manifest.jsonl"),
        json.loads((root / "meta/splits.json").read_text()),
        train_count,
        validation_count,
    )
    manifest = {row["path"]: row for row in read_jsonl(root / "FILE_MANIFEST.jsonl")}
    metadata_artifacts = []
    for filename in METADATA:
        if filename == "FILE_MANIFEST.jsonl":
            continue  # A manifest cannot carry its own digest.
        path = root / filename
        record = manifest.get(filename)
        actual = sha256(path)
        if record is None or record["sha256"] != actual or record["bytes"] != path.stat().st_size:
            raise ValueError(f"metadata checksum or size mismatch: {filename}")
        metadata_artifacts.append({"path": filename, "bytes": path.stat().st_size, "sha256": actual})
    artifacts = []
    for row in selected:
        files = [(row["data_path"], row["parquet_sha256"])] + [
            (path, row["video_sha256"][side]) for side, path in row["video_paths"].items()
        ]
        for filename, expected in files:
            path = Path(
                hf_hub_download(
                    REPO_ID, filename=filename, repo_type="dataset", revision=revision, local_dir=root
                )
            )
            actual = sha256(path)
            record = manifest.get(filename)
            if (
                actual != expected
                or record is None
                or record["sha256"] != actual
                or record["bytes"] != path.stat().st_size
            ):
                raise ValueError(f"checksum or size mismatch: {filename}")
            artifacts.append(
                {
                    "episode_index": row["episode_index"],
                    "split": row["split"],
                    "path": filename,
                    "bytes": path.stat().st_size,
                    "sha256": actual,
                }
            )
            print(f"verified {filename}: {path.stat().st_size} bytes")
    selection = {
        "repo_id": REPO_ID,
        "revision": revision,
        "source_task": SOURCE_TASK,
        "counts": counts,
        "episodes": selected,
        "metadata_artifacts": metadata_artifacts,
        "artifacts": artifacts,
    }
    (root / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--train-count", type=int, default=1)
    parser.add_argument("--validation-count", type=int, default=1)
    parser.add_argument(
        "--all-t46", action="store_true", help="Select all T46 episodes; never the full HandsOn corpus"
    )
    args = parser.parse_args()
    train_count = validation_count = None if args.all_t46 else 1
    if not args.all_t46:
        train_count, validation_count = args.train_count, args.validation_count
    selection = download(args.raw_root, args.revision, train_count, validation_count)
    print(
        f"T46 counts: {selection['counts']}; selected: {[(r['episode_index'], r['split']) for r in selection['episodes']]}"
    )


if __name__ == "__main__":
    main()
