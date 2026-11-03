from pathlib import Path

import draccus
import torch

import lerobot.policies  # noqa: F401 -- register GrootConfig for draccus
from lerobot.configs.train import TrainPipelineConfig
from lerobot.lerobot_types import TransitionKey
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.groot.processor_groot import (
    GrootActionUnpackUnnormalizeStep,
    GrootN17PackInputsStep,
    _transform_n1_7_image_for_vlm_torch,
)

TASK_DIR = Path(__file__).parents[1]


def test_train_and_smoke_configs_parse_with_current_cli_schema():
    for name, steps, batch_size in (("train_config.json", 1000000, 4), ("smoke_config.json", 3, 1)):
        config = draccus.parse(TrainPipelineConfig, args=[f"--config_path={TASK_DIR / name}"])
        assert isinstance(config.policy, GrootConfig)
        assert config.steps == steps and config.batch_size == batch_size
        assert config.policy.chunk_size == 16 and config.policy.use_relative_actions is False
        assert config.policy.input_features["observation.state"].shape == (52,)
        assert config.policy.output_features["action"].shape == (52,)
        assert (
            len([key for key in config.policy.input_features if key.startswith("observation.images.")]) == 3
        )


def test_pack_masks_padding_and_unnormalizes_absolute_52d_actions():
    raw = torch.linspace(0.1, 0.9, 52).repeat(1, 16, 1)
    stats = {key: {"min": [0.0] * 52, "max": [1.0] * 52} for key in ("observation.state", "action")}
    step = GrootN17PackInputsStep(
        action_horizon=16,
        valid_action_horizon=16,
        max_state_dim=132,
        max_action_dim=132,
        stats=stats,
        clip_outliers=False,
    )
    transition = {
        TransitionKey.OBSERVATION: {
            "observation.state": torch.zeros(1, 52),
            "observation.images.front": torch.zeros(1, 3, 480, 854, dtype=torch.uint8),
            "observation.images.wrist_left": torch.zeros(1, 3, 480, 848, dtype=torch.uint8),
            "observation.images.wrist_right": torch.zeros(1, 3, 480, 848, dtype=torch.uint8),
        },
        TransitionKey.ACTION: raw.clone(),
        TransitionKey.COMPLEMENTARY_DATA: {
            "task": ["Store the charger in the drawer."],
            "action_is_pad": torch.tensor([[False] * 15 + [True]]),
        },
    }
    packed = step(transition)
    assert packed[TransitionKey.OBSERVATION]["video"].shape == (1, 1, 3, 480, 854, 3)
    assert packed[TransitionKey.OBSERVATION]["state"].shape == (1, 1, 132)
    assert packed[TransitionKey.ACTION].shape == (1, 16, 132)
    assert packed[TransitionKey.COMPLEMENTARY_DATA]["action_mask"][0, 15].sum() == 0
    assert packed[TransitionKey.COMPLEMENTARY_DATA]["action_mask"][0, 0].sum() == 52
    unpack = GrootActionUnpackUnnormalizeStep(env_action_dim=52, stats=stats)
    decoded = unpack({TransitionKey.ACTION: packed[TransitionKey.ACTION][:, :15]})[TransitionKey.ACTION]
    torch.testing.assert_close(decoded, raw[:, :15], atol=1e-6, rtol=0)
    resized = _transform_n1_7_image_for_vlm_torch(
        torch.zeros(3, 480, 854, dtype=torch.uint8),
        image_crop_size=[230, 230],
        image_target_size=[256, 256],
        shortest_image_edge=None,
        crop_fraction=None,
    )
    assert resized.shape == (3, 256, 256)


def test_observation_only_pack_enables_the_full_inference_horizon():
    step = GrootN17PackInputsStep(
        action_horizon=16,
        valid_action_horizon=16,
        max_state_dim=132,
        max_action_dim=132,
        normalize_min_max=False,
    )
    packed = step(
        {
            TransitionKey.OBSERVATION: {
                "observation.state": torch.zeros(1, 52),
                "observation.images.front": torch.zeros(1, 3, 480, 854, dtype=torch.uint8),
                "observation.images.wrist_left": torch.zeros(1, 3, 480, 848, dtype=torch.uint8),
                "observation.images.wrist_right": torch.zeros(1, 3, 480, 848, dtype=torch.uint8),
            },
            TransitionKey.COMPLEMENTARY_DATA: {"task": ["Store the charger."]},
        }
    )
    mask = packed[TransitionKey.COMPLEMENTARY_DATA]["action_mask"]
    assert mask.shape == (1, 16) and mask.all()
    assert packed.get(TransitionKey.ACTION) is None
