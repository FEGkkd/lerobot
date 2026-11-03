import pyarrow as pa
import pytest

from examples.handson.t46_store_charger.validate_dataset import validate_numeric_episode


def make_episode(action_value=0.0):
    return pa.table(
        {
            "observation.state": [[0.0] * 52 for _ in range(3)],
            "action": [[action_value] * 52 for _ in range(3)],
            "timestamp": [0.0, 1 / 30, 2 / 30],
            "frame_index": [0, 1, 2],
            "episode_index": [7] * 3,
        }
    )


def test_numeric_episode_has_52d_and_30hz():
    result = validate_numeric_episode(make_episode(), 7, 3)
    assert result["frames"] == 3
    assert result["state_shape"] == [3, 52]
    assert result["action_shape"] == [3, 52]


def test_numeric_episode_rejects_nonfinite_action():
    with pytest.raises(ValueError, match="nonfinite"):
        validate_numeric_episode(make_episode(float("nan")), 7, 3)
