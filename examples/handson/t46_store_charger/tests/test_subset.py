from pathlib import Path

import pytest

from examples.handson.t46_store_charger.download_subset import select_episodes
from examples.handson.t46_store_charger.prepare_dataset import remap_parquet, valid_frames


def test_select_episodes_respects_official_split():
    rows = [
        {
            "episode_index": 1,
            "source_task": "bench320_fu006_hold_drawer_store_charger",
            "split": "validation",
            "frames": 10,
        },
        {
            "episode_index": 2,
            "source_task": "bench320_fu006_hold_drawer_store_charger",
            "split": "train",
            "frames": 20,
        },
        {"episode_index": 3, "source_task": "other", "split": "train", "frames": 1},
    ]
    selected, counts = select_episodes(rows, {"train": [2, 3], "validation": [1]}, 1, 1)
    assert counts == {"train": 1, "validation": 1}
    assert [(r["episode_index"], r["split"]) for r in selected] == [(2, "train"), (1, "validation")]


def test_select_episodes_rejects_split_disagreement():
    rows = [
        {
            "episode_index": 1,
            "source_task": "bench320_fu006_hold_drawer_store_charger",
            "split": "train",
            "frames": 10,
        }
    ]
    with pytest.raises(ValueError, match="split disagreement"):
        select_episodes(rows, {"train": [], "validation": [1]}, 1, 0)


def test_remap_parquet_preserves_numeric_values(tmp_path: Path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    src = tmp_path / "source.parquet"
    dst = tmp_path / "out.parquet"
    table = pa.table(
        {
            "observation.state": pa.array([[1.0] * 52, [2.0] * 52]),
            "action": pa.array([[3.0] * 52, [4.0] * 52]),
            "timestamp": [0.0, 1 / 30],
            "frame_index": [0, 1],
            "episode_index": [8375, 8375],
            "index": [100, 101],
            "task_index": [10, 10],
        }
    )
    pq.write_table(table, src)
    remap_parquet(src, dst, episode_index=0, index_offset=0)
    out = pq.read_table(dst)
    assert out["observation.state"].to_pylist() == table["observation.state"].to_pylist()
    assert out["action"].to_pylist() == table["action"].to_pylist()
    assert out["episode_index"].to_pylist() == [0, 0]
    assert out["index"].to_pylist() == [0, 1]
    assert out["task_index"].to_pylist() == [0, 0]


def test_incomplete_command_tail_is_excluded(tmp_path: Path):
    assert (
        valid_frames({"frames": 10, "active_sides": ["right", "left"], "pose_commanded_counts": [8, 8]}) == 8
    )
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    source = tmp_path / "source.parquet"
    target = tmp_path / "target.parquet"
    pq.write_table(
        pa.table(
            {
                "observation.state": [[0.0] * 52] * 3,
                "action": [[1.0] * 52] * 3,
                "timestamp": [0.0, 1 / 30, 2 / 30],
                "frame_index": [0, 1, 2],
                "episode_index": [7] * 3,
                "index": [4, 5, 6],
                "task_index": [2] * 3,
            }
        ),
        source,
    )
    assert remap_parquet(source, target, 0, 0, valid_length=2) == 2
    assert pq.read_table(target).num_rows == 2
