"""Validate explicit episode partitions and the provenance of their normalization stats."""

import hashlib
import json
from pathlib import Path
from typing import Any


def load_episode_splits(
    path: str | Path, repo_id: str, revision: str | None, total_episodes: int
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    manifest_path = Path(path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest["source"]
    if source["repo_id"] != repo_id or source["revision"] != revision:
        raise ValueError("Split manifest does not match the configured dataset revision.")
    splits = manifest["splits"]
    seen: set[int] = set()
    for name in ("train", "validation", "test"):
        indices = splits[name]
        if not indices or any(type(i) is not int or not 0 <= i < total_episodes for i in indices):
            raise ValueError(f"Invalid or empty {name} episode partition.")
        if len(indices) != len(set(indices)) or seen.intersection(indices):
            raise ValueError("Episode partitions overlap or contain duplicates.")
        seen.update(indices)
    if seen != set(range(total_episodes)):
        raise ValueError("Episode partitions must cover the dataset exactly once.")
    statistics = manifest["statistics"]
    if statistics["source_split"] != "train" or statistics["episodes"] != splits["train"]:
        raise ValueError("Normalization statistics must come from exactly the training episodes.")
    stats_path = (manifest_path.parent / statistics["path"]).resolve()
    if not stats_path.is_relative_to(manifest_path.parent):
        raise ValueError("Statistics must be stored beside the split manifest.")
    content = stats_path.read_bytes()
    if hashlib.sha256(content).hexdigest() != statistics["sha256"]:
        raise ValueError("Training statistics checksum mismatch.")
    return splits, json.loads(content)
