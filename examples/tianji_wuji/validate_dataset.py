#!/usr/bin/env python
"""Check the original Tianji/Wuji LeRobot v3.0 dataset used by the GR00T task."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset

CAMERAS = {
    "observation.images.head_left": (3, 640, 640),
    "observation.images.head_right": (3, 640, 640),
    "observation.images.wrist_left": (3, 480, 640),
    "observation.images.wrist_right": (3, 480, 640),
}
DATASET_ROOT = "/home/ubuntu/MANUS_STEAMVR_ws/data_process/datasets/lerobot_bag_batter_head_stereo_640"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(DATASET_ROOT))
    args = parser.parse_args()
    root = args.root
    info = json.loads((root / "meta/info.json").read_text())
    assert info["codebase_version"] == "v3.0", info["codebase_version"]
    assert info["total_episodes"] == 110, info["total_episodes"]
    assert info["total_frames"] == 41399, info["total_frames"]
    assert info["fps"] == 30, info["fps"]
    process = json.loads((root / "meta/data_process.json").read_text())
    assert len(process["episodes"]) == 110
    assert all(episode["quality_status"] == "accepted" for episode in process["episodes"])
    assert all(episode["task"] == "bag_batter" for episode in process["episodes"])
    assert sum(episode["dropped_frame_count"] for episode in process["episodes"]) == 0
    assert info["features"]["observation.state"]["shape"] == [54]
    assert info["features"]["action"]["shape"] == [54]
    assert info["features"]["action"]["names"] == info["features"]["observation.state"]["names"]
    stats = json.loads((root / "meta/stats.json").read_text())
    for key in ("observation.state", "action"):
        assert len(stats[key]["min"]) == 54
        assert len(stats[key]["max"]) == 54
    assert {key: tuple(info["features"][key]["shape"]) for key in CAMERAS} == CAMERAS
    episodes = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet")
    assert episodes.num_rows == 110
    assert sum(episodes["length"].to_pylist()) == 41399
    data_chunks = episodes["data/chunk_index"].to_pylist()
    data_files = episodes["data/file_index"].to_pylist()
    for chunk_index, file_index in set(zip(data_chunks, data_files, strict=True)):
        data_path = root / f"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
        assert data_path.is_file(), data_path
        table = pq.read_table(data_path, columns=["observation.state", "action"])
        for key in ("observation.state", "action"):
            values = table[key].combine_chunks().values.to_numpy().reshape(-1, 54)
            assert np.isfinite(values).all(), (data_path, key)
    for key in CAMERAS:
        chunk_indices = episodes[f"videos/{key}/chunk_index"].to_pylist()
        file_indices = episodes[f"videos/{key}/file_index"].to_pylist()
        for chunk_index, file_index in set(zip(chunk_indices, file_indices, strict=True)):
            video_path = root / f"videos/{key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
            assert video_path.is_file(), video_path
    meta = LeRobotDatasetMetadata("fegkkd/bag_batter_head_stereo_640", root=root)
    assert set(meta.tasks.index) == {"bag_batter"}, list(meta.tasks.index)
    delta_timestamps = {"action": [i / meta.fps for i in range(16)]}
    dataset = LeRobotDataset(
        "fegkkd/bag_batter_head_stereo_640",
        root=root,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
        return_uint8=True,
    )
    for index in (0, len(dataset) // 2, len(dataset) - 16):
        item = dataset[index]
        assert item["task"] == "bag_batter", (index, item["task"])
        assert tuple(item["observation.state"].shape) == (54,)
        assert tuple(item["action"].shape) == (16, 54)
        assert torch.isfinite(item["observation.state"]).all()
        assert torch.isfinite(item["action"]).all()
        for key, shape in CAMERAS.items():
            assert tuple(item[key].shape) == shape, (index, key, item[key].shape)
    print(
        f"OK: {meta.total_episodes} episodes, {meta.total_frames} frames, "
        "four cameras, 54D state/action, 16-step action windows"
    )


if __name__ == "__main__":
    main()
