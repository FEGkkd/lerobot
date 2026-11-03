from pathlib import Path

from examples.handson.t46_store_charger.cluster_check import checkpoint_directory, observation_only_batch
from examples.handson.t46_store_charger.run_long import stop_reason


def test_million_step_resume_paths_use_native_seven_digit_width():
    root = Path("train")
    assert checkpoint_directory(root, 3) == root / "checkpoints/0000003"
    assert checkpoint_directory(root, 500) == root / "checkpoints/0000500"
    assert checkpoint_directory(root, 1000000) == root / "checkpoints/1000000"


def test_phase_controls_preserve_manual_stop_and_resource_limits(tmp_path):
    ample_storage = 200 * 1024**3
    assert stop_reason(tmp_path, 7200, 600, ample_storage) is None
    assert stop_reason(tmp_path, 1799, 600, ample_storage) == "runtime_budget"
    assert stop_reason(tmp_path, 3000, 2000, ample_storage) == "runtime_budget"
    assert stop_reason(tmp_path, 7200, 600, 59 * 1024**3) == "storage_budget"
    (tmp_path / "STOP_AFTER_PHASE").touch()
    assert stop_reason(tmp_path, 7200, 600, ample_storage) == "user_stop"


def test_inference_observations_do_not_include_dataset_actions():
    batch = {
        "observation.state": [0],
        "observation.images.front": [1],
        "task": "store",
        "action": [999],
        "action_is_pad": [False],
        "index": 3,
    }
    assert set(observation_only_batch(batch)) == {"observation.state", "observation.images.front", "task"}
    assert "action" in batch
