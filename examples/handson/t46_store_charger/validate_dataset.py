"""Validate real T46 source files, v3 conversion, video timing, and DataLoader."""

import argparse
import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq

from examples.handson.t46_store_charger.download_subset import DEFAULT_ROOT, sha256
from examples.handson.t46_store_charger.prepare_dataset import DEFAULT_OUTPUT, valid_frames


def validate_numeric_episode(table, episode_id: int, expected_frames: int) -> dict:
    if table.num_rows != expected_frames:
        raise ValueError(f"frame count mismatch for episode {episode_id}")
    if table["episode_index"].to_pylist() != [episode_id] * expected_frames:
        raise ValueError(f"episode boundary mismatch for episode {episode_id}")
    if table["frame_index"].to_pylist() != list(range(expected_frames)):
        raise ValueError(f"frame index mismatch for episode {episode_id}")
    result = {"frames": expected_frames}
    for key, label in (("observation.state", "state"), ("action", "action")):
        values = np.asarray(table[key].to_pylist(), dtype=np.float32)
        if values.shape != (expected_frames, 52):
            raise ValueError(f"{label} shape mismatch: {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite {label} values in episode {episode_id}")
        result[f"{label}_shape"] = list(values.shape)
    timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
    max_error = float(np.max(np.abs(timestamps - np.arange(expected_frames) / 30)))
    if max_error > 1e-4:
        raise ValueError(f"30 Hz timestamp mismatch: {max_error}")
    result["timestamp_max_error_s"] = max_error
    return result


def validate_video(path: Path, frames: int | None, shape: tuple[int, int]) -> dict:
    count = 0
    max_error = 0.0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(video=0):
            image = frame.to_ndarray(format="rgb24")
            if image.shape != (*shape, 3):
                raise ValueError(f"image shape mismatch in {path}: {image.shape}")
            if frame.pts is None:
                raise ValueError(f"missing video timestamp in {path}")
            stamp = float(frame.pts * stream.time_base)
            max_error = max(max_error, abs(stamp - count / 30))
            count += 1
    if (frames is not None and count != frames) or max_error > 1 / 60:
        raise ValueError(f"video frames/timing mismatch in {path}: {count} vs {frames}, error {max_error}")
    return {"frames": count, "height": shape[0], "width": shape[1], "timestamp_max_error_s": max_error}


def validate(raw_root: Path, output: Path) -> dict:
    selection = json.loads((raw_root / "selection.json").read_text())
    provenance = json.loads((output / "meta/source_provenance.json").read_text())
    if provenance["revision"] != selection["revision"]:
        raise ValueError("revision mismatch")
    schema = provenance["policy_schema"]
    if (
        schema["state_dim"],
        schema["action_dim"],
        schema["fps"],
        schema["side_order"],
        schema["action_semantics"],
        schema["use_delta_state_actions"],
    ) != (52, 52, 30, ["right", "left"], "absolute_controller_command", False):
        raise ValueError("HandsOn 52D absolute-action schema mismatch")
    if [row["source_episode_index"] for row in provenance["episodes"]] != [
        row["episode_index"] for row in selection["episodes"]
    ]:
        raise ValueError("episode provenance mismatch")
    source_results = []
    raw_tables = []
    total_frames = 0
    for row in selection["episodes"]:
        source = raw_root / row["data_path"]
        if sha256(source) != row["parquet_sha256"]:
            raise ValueError(f"source Parquet checksum mismatch: {source}")
        table = pq.read_table(source)
        numeric = validate_numeric_episode(table, row["episode_index"], row["frames"])
        valid_length = valid_frames(row)
        if any("pad" in key or "mask" in key for key in table.column_names):
            raise ValueError(
                "unexpected raw padding/active mask column; conversion needs an explicit mapping"
            )
        videos = {}
        for side, relpath in row["video_paths"].items():
            path = raw_root / relpath
            if sha256(path) != row["video_sha256"][side] or path.stat().st_size != row["video_bytes"][side]:
                raise ValueError(f"source video checksum/size mismatch: {path}")
            videos[side] = validate_video(path, row["frames"], (480, 854 if side == "front" else 848))
        source_results.append(
            {"episode_index": row["episode_index"], "split": row["split"], **numeric, "videos": videos}
        )
        raw_tables.append(table)
        total_frames += valid_length

    for side in ("front", "wrist_left", "wrist_right"):
        video_files = sorted((output / "videos" / f"observation.images.{side}").glob("*/*.mp4"))
        decoded = sum(
            validate_video(path, None, (480, 854 if side == "front" else 848))["frames"]
            for path in video_files
        )
        if decoded != total_frames:
            raise ValueError(f"v3 {side} frame count mismatch: {decoded} vs {total_frames}")

    v3_parts = [pq.read_table(path) for path in sorted((output / "data").glob("*/*.parquet"))]
    import pyarrow as pa

    v3 = pa.concat_tables(v3_parts)
    if v3.num_rows != total_frames:
        raise ValueError("v3 total frame count mismatch")
    offset = 0
    for local_index, (source, row) in enumerate(zip(raw_tables, selection["episodes"], strict=True)):
        length = valid_frames(row)
        converted = v3.slice(offset, length)
        validate_numeric_episode(converted, local_index, length)
        for key in ("observation.state", "action", "timestamp"):
            if not np.array_equal(
                np.asarray(source[key].to_pylist()[:length]), np.asarray(converted[key].to_pylist())
            ):
                raise ValueError(f"numeric mismatch for {key}, episode {row['episode_index']}")
        offset += length

    from torch.utils.data import DataLoader

    from lerobot.datasets import EpisodeAwareSampler, LeRobotDataset
    from lerobot.datasets.episode_splits import load_episode_splits

    splits, stats = load_episode_splits(
        output / "meta/split_manifest.json", "handson/t46_store_charger", None, len(selection["episodes"])
    )
    if set(splits["train"]) & set(splits["validation"]) or "q01" not in stats["action"]:
        raise ValueError("split leakage or missing train quantiles")
    for key in ("observation.state", "action"):
        train_values = np.concatenate(
            [
                np.asarray(raw_tables[index][key].to_pylist()[: valid_frames(selection["episodes"][index])])
                for index in splits["train"]
            ]
        )
        for stat, expected in (
            ("mean", train_values.mean(axis=0)),
            ("q01", np.quantile(train_values, 0.01, axis=0)),
            ("q99", np.quantile(train_values, 0.99, axis=0)),
        ):
            if not np.allclose(stats[key][stat], expected, atol=1e-5, rtol=0):
                raise ValueError(f"train-only statistic mismatch: {key}/{stat}")
    train = LeRobotDataset(
        "handson/t46_store_charger",
        root=output,
        episodes=splits["train"],
        revision=None,
        video_backend="pyav",
        return_uint8=True,
        delta_timestamps={"action": [i / 30 for i in range(16)]},
    )
    valid = LeRobotDataset(
        "handson/t46_store_charger",
        root=output,
        episodes=splits["validation"],
        revision=None,
        video_backend="pyav",
        return_uint8=True,
        delta_timestamps={"action": [i / 30 for i in range(16)]},
    )
    train_length = sum(valid_frames(row) for row in selection["episodes"] if row["split"] == "train")
    valid_length = sum(valid_frames(row) for row in selection["episodes"] if row["split"] == "validation")
    if (
        train.meta.total_episodes != len(selection["episodes"])
        or len(train) != train_length
        or len(valid) != valid_length
    ):
        raise ValueError("LeRobotDataset episode boundaries mismatch")
    sampler = EpisodeAwareSampler(
        train.meta.episodes["dataset_from_index"],
        train.meta.episodes["dataset_to_index"],
        episode_indices_to_use=train.episodes,
        drop_n_last_frames=15,
        absolute_to_relative_idx=train.absolute_to_relative_idx,
    )
    batch = next(iter(DataLoader(train, batch_size=2, sampler=sampler, num_workers=0)))
    expected_images = {
        "front": (2, 3, 480, 854),
        "wrist_left": (2, 3, 480, 848),
        "wrist_right": (2, 3, 480, 848),
    }
    if (
        tuple(batch["observation.state"].shape) != (2, 52)
        or tuple(batch["action"].shape) != (2, 16, 52)
        or batch["action_is_pad"].any()
    ):
        raise ValueError("training batch state/action shape or padding mismatch")
    for side, shape in expected_images.items():
        if tuple(batch[f"observation.images.{side}"].shape) != shape:
            raise ValueError(f"training image shape mismatch: {side}")
    from lerobot.lerobot_types import TransitionKey
    from lerobot.policies.groot.processor_groot import (
        GrootActionUnpackUnnormalizeStep,
        GrootN17PackInputsStep,
        _transform_n1_7_image_for_vlm_torch,
    )

    raw_action = batch["action"].clone()
    packed = GrootN17PackInputsStep(
        action_horizon=16,
        valid_action_horizon=16,
        max_state_dim=132,
        max_action_dim=132,
        stats=stats,
        clip_outliers=False,
    )(
        {
            TransitionKey.OBSERVATION: {
                key: value for key, value in batch.items() if key.startswith("observation.")
            },
            TransitionKey.ACTION: raw_action.clone(),
            TransitionKey.COMPLEMENTARY_DATA: {
                "task": batch["task"],
                "action_is_pad": batch["action_is_pad"],
            },
        }
    )
    action_mask = packed[TransitionKey.COMPLEMENTARY_DATA]["action_mask"]
    if (
        tuple(action_mask.shape) != (2, 16, 132)
        or not np.all(action_mask[:, :, :52].cpu().numpy() == 1)
        or np.any(action_mask[:, :, 52:].cpu().numpy())
    ):
        raise ValueError("GR00T action dimension/padding mask mismatch")
    unpacked = GrootActionUnpackUnnormalizeStep(env_action_dim=52, stats=stats)(
        {TransitionKey.ACTION: packed[TransitionKey.ACTION]}
    )[TransitionKey.ACTION]
    if not np.allclose(unpacked.cpu().numpy(), raw_action.cpu().numpy(), rtol=0, atol=2e-5):
        raise ValueError("GR00T absolute action normalization round-trip mismatch")
    resized = _transform_n1_7_image_for_vlm_torch(
        batch["observation.images.front"][0],
        image_crop_size=[230, 230],
        image_target_size=[256, 256],
        shortest_image_edge=None,
        crop_fraction=None,
    )
    if tuple(resized.shape) != (3, 256, 256):
        raise ValueError("GR00T image resize mismatch")
    last_train = train[len(train) - 1]
    if not last_train["action_is_pad"].any() or len(sampler) != len(train) - 15 * len(splits["train"]):
        raise ValueError("episode-tail padding mask or sampler exclusion mismatch")
    result = {
        "revision": selection["revision"],
        "source_episodes": source_results,
        "v3_root": str(output),
        "v3_frames": total_frames,
        "splits": splits,
        "train_sampler_frames": len(sampler),
        "batch_state_shape": list(batch["observation.state"].shape),
        "batch_action_shape": list(batch["action"].shape),
        "batch_image_shapes": {key: list(shape) for key, shape in expected_images.items()},
        "numeric_equal": True,
        "train_statistics_match_raw": True,
        "video_full_decode": True,
        "groot_pack_action_mask": list(action_mask.shape),
        "groot_absolute_action_roundtrip": True,
        "groot_image_preprocess_shape": list(resized.shape),
    }
    (output / "validation_report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--v3-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(validate(args.raw_root, args.v3_root), indent=2))


if __name__ == "__main__":
    main()
