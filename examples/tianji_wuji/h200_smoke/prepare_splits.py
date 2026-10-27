"""Generate immutable episode partitions and exact training-only joint statistics."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

DATASET_REVISION = "e82d64b44deff9dc4d42c149e6048010a5427ccd"
SOURCE_CODE_REVISION = "9c3dbbd3203ce0d1192ad9150e90f2901f29dd94"
MODEL_REVISIONS = {
    "nvidia/GR00T-N1.7-3B": "2fc962b973bccdd5d8ce4f67cc63b264d6886495",
    "nvidia/Cosmos-Reason2-2B": "9ce19a195e423419c349abfc86fd07178b230561",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def partition(source_ids: list[str], seed: int = 42) -> dict[str, list[int]]:
    if len(source_ids) != 110 or len(set(source_ids)) != 110:
        raise ValueError("Expected 110 distinct original demonstration IDs.")
    ranked = sorted(
        range(len(source_ids)),
        key=lambda index: hashlib.sha256(f"{seed}:{source_ids[index]}".encode()).hexdigest(),
    )
    return {
        "train": sorted(ranked[:88]),
        "validation": sorted(ranked[88:99]),
        "test": sorted(ranked[99:]),
    }


def joint_statistics(values: np.ndarray) -> dict[str, list]:
    if values.ndim != 2 or values.shape[1] != 54 or not np.isfinite(values).all():
        raise ValueError("Expected finite 54-dimensional joint vectors.")
    if not len(values):
        raise ValueError("Training partition contains no samples.")
    values = values.astype(np.float64)
    result = {
        "count": [len(values)],
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
    }
    for quantile in (1, 10, 50, 90, 99):
        result[f"q{quantile:02d}"] = np.quantile(values, quantile / 100, axis=0).tolist()
    return result


def prepare(root: Path, destination: Path) -> dict:
    info = json.loads((root / "meta/info.json").read_text())
    if (info["codebase_version"], info["total_episodes"], info["total_frames"], info["fps"]) != (
        "v3.0",
        110,
        41399,
        30,
    ):
        raise ValueError("Unexpected source dataset metadata.")
    process = json.loads((root / "meta/data_process.json").read_text())
    episodes = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    if [row["episode_index"] for row in episodes] != list(range(110)):
        raise ValueError("Unexpected episode ordering.")
    if [row["length"] for row in episodes] != [row["frame_count"] for row in process["episodes"]]:
        raise ValueError("Original demonstration IDs do not align with episode metadata.")
    source_ids = [row["source_episode_id"] for row in process["episodes"]]
    splits = partition(source_ids)
    data_files = sorted(
        {
            f"data/chunk-{row['data/chunk_index']:03d}/file-{row['data/file_index']:03d}.parquet"
            for row in episodes
        }
    )
    columns = ("observation.state", "action")
    training_values: dict[str, list[np.ndarray]] = {key: [] for key in columns}
    frame_counts = dict.fromkeys(splits, 0)
    for relative in data_files:
        table = pq.read_table(root / relative, columns=["episode_index", *columns])
        indices = table["episode_index"].to_numpy()
        train_mask = np.isin(indices, splits["train"])
        for name, ids in splits.items():
            frame_counts[name] += int(np.isin(indices, ids).sum())
        for key in columns:
            values = np.asarray(table[key].to_pylist(), dtype=np.float64)
            if values.shape != (len(indices), 54) or not np.isfinite(values).all():
                raise ValueError(f"Invalid {key} in {relative}")
            training_values[key].append(values[train_mask])
    expected_counts = {name: sum(episodes[index]["length"] for index in ids) for name, ids in splits.items()}
    if frame_counts != expected_counts or sum(frame_counts.values()) != 41399:
        raise ValueError("Episode rows do not match the frame metadata.")
    statistics = {key: joint_statistics(np.concatenate(values)) for key, values in training_values.items()}
    stats_bytes = encode(statistics)
    manifest = {
        "format_version": 1,
        "source": {
            "repo_id": "fegkkd/bag_v3",
            "revision": DATASET_REVISION,
            "files_sha256": {
                relative: sha256_file(root / relative)
                for relative in ["meta/info.json", "meta/data_process.json", *data_files]
            },
        },
        "method": "sha256(seed:source_episode_id), whole episodes; no session grouping metadata available",
        "seed": 42,
        "splits": splits,
        "frame_counts": frame_counts,
        "source_episode_ids": source_ids,
        "statistics": {
            "path": "train_stats.json",
            "sha256": hashlib.sha256(stats_bytes).hexdigest(),
            "source_split": "train",
            "episodes": splits["train"],
            "frames": frame_counts["train"],
        },
    }
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in (("train_stats.json", stats_bytes), ("split_manifest.json", encode(manifest))):
        target = destination / name
        if target.exists() and target.read_bytes() != content:
            raise FileExistsError(f"Refusing to replace a different immutable artifact: {target}")
        target.write_bytes(content)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "splits")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        manifest = json.loads((args.output / "split_manifest.json").read_text())
        for relative, expected in manifest["source"]["files_sha256"].items():
            if sha256_file(args.root / relative) != expected:
                raise ValueError(f"Source checksum mismatch: {relative}")
        if sha256_file(args.output / "train_stats.json") != manifest["statistics"]["sha256"]:
            raise ValueError("Training statistics checksum mismatch.")
        print("Pinned source and immutable split/statistics checksums verified.")
        raise SystemExit(0)
    result = prepare(args.root, args.output)
    print(
        json.dumps(
            {
                "episodes": {key: len(value) for key, value in result["splits"].items()},
                "frames": result["frame_counts"],
            },
            indent=2,
        )
    )
