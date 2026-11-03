import hashlib
import json

from lerobot.datasets.episode_splits import load_episode_splits


def test_two_episode_official_split_has_no_test_partition(tmp_path):
    stats = b'{"action":{"mean":[0]}}\n'
    (tmp_path / "train_stats.json").write_bytes(stats)
    manifest = {
        "source": {"repo_id": "handson/t46_store_charger", "revision": None},
        "splits": {"train": [0], "validation": [1], "test": []},
        "statistics": {
            "source_split": "train",
            "episodes": [0],
            "path": "train_stats.json",
            "sha256": hashlib.sha256(stats).hexdigest(),
        },
    }
    (tmp_path / "split_manifest.json").write_text(json.dumps(manifest))
    splits, _ = load_episode_splits(tmp_path / "split_manifest.json", "handson/t46_store_charger", None, 2)
    assert splits == {"train": [0], "validation": [1], "test": []}
