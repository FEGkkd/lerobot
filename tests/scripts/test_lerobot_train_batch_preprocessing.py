# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.scripts.lerobot_train import _preprocess_dataset_batch  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_IMAGES  # noqa: E402


def test_preprocess_dataset_batch_normalizes_and_renames_before_processor():
    seen_batch = None

    def preprocessor(batch):
        nonlocal seen_batch
        seen_batch = batch
        return batch

    batch = {
        "image": torch.full((1, 3, 2, 2), 255, dtype=torch.uint8),
        "actions": torch.ones(1, 2),
    }
    processed = _preprocess_dataset_batch(
        batch,
        camera_keys=["image"],
        rename_map={"image": f"{OBS_IMAGES}.camera1", "actions": ACTION},
        preprocessor=preprocessor,
    )

    assert seen_batch is processed
    assert set(processed) == {f"{OBS_IMAGES}.camera1", ACTION}
    assert processed[f"{OBS_IMAGES}.camera1"].dtype == torch.float32
    torch.testing.assert_close(processed[f"{OBS_IMAGES}.camera1"], torch.ones(1, 3, 2, 2))


def test_preprocess_dataset_batch_maps_task_prompt_before_processor():
    batch = {"task": ["bag_batter", "other_task"], "observation.teleop_input": torch.zeros(2, 22)}
    processed = _preprocess_dataset_batch(
        batch,
        camera_keys=[],
        rename_map={},
        preprocessor=lambda value: value,
        task_prompt_map={"bag_batter": "Put the drink bottle into the bag."},
        exclude_features=["observation.teleop_input"],
    )
    assert processed["task"] == ["Put the drink bottle into the bag.", "other_task"]
    assert "observation.teleop_input" not in processed
