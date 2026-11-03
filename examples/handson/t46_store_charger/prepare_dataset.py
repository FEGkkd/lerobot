"""Build a sparse v2.1 subset and run LeRobot's official v2.1-to-v3 converter."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from examples.handson.t46_store_charger.download_subset import DEFAULT_ROOT, read_jsonl, sha256

DEFAULT_OUTPUT = Path("/home/ubuntu/lerobot_ws/datasets/handson_t46/v3")
TEST_SESSION = "20260903-112559_dx_bench320_fu006_hold_drawer_store_charger"


def assigned_splits(episodes: list[dict], test_sessions: tuple[str, ...]) -> list[str]:
    """Hold out complete official-training sessions without moving validation episodes."""
    requested = set(test_sessions)
    known = {row["collection_session_id"] for row in episodes}
    if len(requested) != len(test_sessions) or not requested <= known:
        raise ValueError("Unknown or duplicate held-out session")
    if any(row["split"] != "train" and row["collection_session_id"] in requested for row in episodes):
        raise ValueError("Only official training sessions may become test data")
    result = ["test" if row["collection_session_id"] in requested else row["split"] for row in episodes]
    sessions = {
        name: {
            row["collection_session_id"] for row, split in zip(episodes, result, strict=True) if split == name
        }
        for name in ("train", "validation", "test")
    }
    if any(
        sessions[a] & sessions[b]
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("A collection session crosses split boundaries")
    if not sessions["train"] or not sessions["validation"]:
        raise ValueError("Training and validation must remain nonempty")
    return result


def valid_frames(row: dict) -> int:
    if row["active_sides"] != ["right", "left"]:
        raise ValueError(f"Unexpected active sides in episode {row['episode_index']}")
    counts = row["pose_commanded_counts"]
    if len(counts) != 2 or counts[0] != counts[1] or not 0 < counts[0] <= row["frames"]:
        raise ValueError(f"Unsupported command coverage in episode {row['episode_index']}: {counts}")
    if counts != [row["frames"], row["frames"]]:
        raise ValueError(
            f"Incomplete command coverage in episode {row['episode_index']}; "
            "counts do not establish missing-frame locations. Quarantine this episode."
        )
    return counts[0]


def remap_parquet(
    source: Path, destination: Path, episode_index: int, index_offset: int, valid_length: int | None = None
) -> int:
    table = pq.read_table(source)
    if table["frame_index"].to_pylist() != list(range(table.num_rows)):
        raise ValueError(f"Noncontiguous source frame indices: {source}")
    if valid_length is not None:
        if not 0 < valid_length <= table.num_rows:
            raise ValueError(f"Invalid valid length {valid_length} for {source}")
        table = table.slice(0, valid_length)
    length = table.num_rows
    for key, values in (
        ("episode_index", [episode_index] * length),
        ("index", list(range(index_offset, index_offset + length))),
        ("task_index", [0] * length),
    ):
        position = table.schema.get_field_index(key)
        table = table.set_column(position, key, pa.array(values, type=table.schema.field(key).type))
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, destination)
    return length


def trim_video(source: Path, target: Path, length: int) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-frames:v",
            str(length),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(target),
        ],
        check=True,
    )


def _numeric_stats(table) -> dict:
    result = {}
    for key in ("observation.state", "action"):
        values = np.asarray(table[key].to_pylist(), dtype=np.float32)
        result[key] = {
            "mean": values.mean(axis=0).tolist(),
            "std": values.std(axis=0).tolist(),
            "min": values.min(axis=0).tolist(),
            "max": values.max(axis=0).tolist(),
            "count": [len(values)],
        }
    return result


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def prepare(raw_root: Path, output: Path, force: bool = False, test_sessions: tuple[str, ...] = ()) -> dict:
    if output.exists():
        if not force:
            raise FileExistsError(f"{output} exists; pass --force to replace this generated subset")
        shutil.rmtree(output)
    selection = json.loads((raw_root / "selection.json").read_text())
    episodes = selection["episodes"]
    if not episodes or len({row["episode_index"] for row in episodes}) != len(episodes):
        raise ValueError("No episodes or duplicate source episode IDs")
    memberships = assigned_splits(episodes, test_sessions)
    official_episodes = {row["episode_index"]: row for row in read_jsonl(raw_root / "meta/episodes.jsonl")}
    official_stats = {row["episode_index"]: row for row in read_jsonl(raw_root / "meta/episodes_stats.jsonl")}
    info = json.loads((raw_root / "meta/info.json").read_text())
    if info["codebase_version"] != "v2.1":
        raise ValueError("Expected a LeRobot v2.1 source")
    schema = json.loads((raw_root / "meta/handson_policy_schema.json").read_text())
    if (
        schema["state_dim"],
        schema["action_dim"],
        schema["fps"],
        schema["side_order"],
        schema["action_semantics"],
    ) != (52, 52, 30, ["right", "left"], "absolute_controller_command"):
        raise ValueError("Unexpected HandsOn world-command schema")
    stage = output.parent / f"{output.name}_stage"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    episode_records, stats_records, provenance = [], [], []
    offset = 0
    for local_index, row in enumerate(episodes):
        source_id = row["episode_index"]
        source_path = raw_root / row["data_path"]
        if sha256(source_path) != row["parquet_sha256"]:
            raise ValueError(f"Parquet checksum mismatch for episode {source_id}")
        target_parquet = stage / f"data/chunk-{local_index // 1000:03d}/episode_{local_index:06d}.parquet"
        length = remap_parquet(source_path, target_parquet, local_index, offset, valid_frames(row))
        if row["frames"] != official_episodes[source_id]["length"]:
            raise ValueError(f"Frame count mismatch for episode {source_id}")
        for side, source_video in row["video_paths"].items():
            source = raw_root / source_video
            if sha256(source) != row["video_sha256"][side]:
                raise ValueError(f"Video checksum mismatch for episode {source_id}, {side}")
            target = (
                stage
                / f"videos/chunk-{local_index // 1000:03d}/observation.images.{side}/episode_{local_index:06d}.mp4"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            if length == row["frames"]:
                target.symlink_to(source)
            else:
                trim_video(source, target, length)
        episode_records.append({"episode_index": local_index, "tasks": [row["prompt"]], "length": length})
        stats = (
            official_stats[source_id]["stats"]
            if length == row["frames"]
            else _numeric_stats(pq.read_table(target_parquet))
        )
        stats_records.append({"episode_index": local_index, "stats": stats})
        provenance.append(
            {
                "local_episode_index": local_index,
                "source_episode_index": source_id,
                "episode_uid": row["episode_uid"],
                "source_task": row["source_task"],
                "source_split": row["split"],
                "split": memberships[local_index],
                "prompt": row["prompt"],
                "collection_session_id": row["collection_session_id"],
                "source_frames": row["frames"],
                "valid_frames": length,
                "source_files": [row["data_path"], *row["video_paths"].values()],
            }
        )
        offset += length
    info.update(
        {
            "total_episodes": len(episodes),
            "total_frames": offset,
            "total_tasks": 1,
            "total_videos": len(episodes) * 3,
            "total_chunks": (len(episodes) + 999) // 1000,
            "splits": {"all": f"0:{len(episodes)}"},
        }
    )
    _write_jsonl(stage / "meta/episodes.jsonl", episode_records)
    _write_jsonl(stage / "meta/episodes_stats.jsonl", stats_records)
    _write_jsonl(stage / "meta/tasks.jsonl", [{"task_index": 0, "task": episodes[0]["prompt"]}])
    (stage / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")

    from lerobot.scripts.convert_dataset_v21_to_v30 import convert_dataset

    convert_dataset(repo_id="handson/t46_store_charger", root=stage, push_to_hub=False)
    stage.rename(output)
    shutil.rmtree(stage.parent / f"{stage.name}_old")
    (output / "meta/source_provenance.json").write_text(
        json.dumps(
            {
                "repo_id": selection["repo_id"],
                "revision": selection["revision"],
                "excluded_training_episodes": selection.get("excluded_training_episodes", []),
                "exclusion_reason": selection.get("exclusion_reason"),
                "test_sessions": list(test_sessions),
                "policy_schema": schema,
                "episodes": provenance,
            },
            indent=2,
        )
        + "\n"
    )
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.io_utils import cast_stats_to_numpy

    train_stats = [
        cast_stats_to_numpy(row["stats"])
        for row, split in zip(stats_records, memberships, strict=True)
        if split == "train"
    ]
    if not train_stats:
        raise ValueError("No training episodes to compute statistics")
    numeric_stats = aggregate_stats(train_stats)
    stats_serialized = {
        key: {stat: np.asarray(value).tolist() for stat, value in entry.items()}
        for key, entry in numeric_stats.items()
    }
    # Official v2.1 episode summaries omit quantiles. GR00T's min/max processor
    # needs train-only q01/q99, so calculate exact quantiles from train frames.
    for key in ("observation.state", "action"):
        values = np.concatenate(
            [
                np.asarray(
                    pq.read_table(raw_root / row["data_path"], columns=[key])[key].to_pylist()[
                        : valid_frames(row)
                    ],
                    dtype=np.float32,
                )
                for row, split in zip(episodes, memberships, strict=True)
                if split == "train"
            ]
        )
        for quantile in (1, 10, 50, 90, 99):
            stats_serialized[key][f"q{quantile:02d}"] = (
                np.quantile(values, quantile / 100, axis=0).astype(np.float32).tolist()
            )
    stats_path = output / "meta/train_stats.json"
    stats_path.write_text(json.dumps(stats_serialized, indent=2, sort_keys=True) + "\n")
    splits = {
        name: [i for i, split in enumerate(memberships) if split == name]
        for name in ("train", "validation", "test")
    }
    split_manifest = {
        "source": {
            "repo_id": "handson/t46_store_charger",
            "revision": None,
            "hands_on_revision": selection["revision"],
        },
        "splits": splits,
        "test_sessions": list(test_sessions),
        "statistics": {
            "source_split": "train",
            "episodes": splits["train"],
            "path": "train_stats.json",
            "sha256": sha256(stats_path),
        },
    }
    (output / "meta/split_manifest.json").write_text(json.dumps(split_manifest, indent=2) + "\n")
    print(f"Converted {len(episodes)} episodes, {offset} frames to {output}")
    return split_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--test-session", action="append", default=[])
    args = parser.parse_args()
    prepare(args.raw_root, args.output, args.force, tuple(args.test_session))


if __name__ == "__main__":
    main()
