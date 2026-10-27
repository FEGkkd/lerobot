"""CPU tests for data isolation, metrics and checkpoint acceptance; no torch required."""

import ast
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
TASK = ROOT / "examples/tianji_wuji/h200_smoke"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


prepare = module("prepare_splits", TASK / "prepare_splits.py")
splits_module = module("episode_splits", ROOT / "src/lerobot/datasets/episode_splits.py")
metrics_module = module("local_metrics", ROOT / "src/lerobot/utils/local_metrics.py")


def test_checked_in_split_has_exact_coverage_and_training_only_statistics():
    manifest_path = TASK / "splits/split_manifest.json"
    partitions, stats = splits_module.load_episode_splits(
        manifest_path, "fegkkd/bag_v3", prepare.DATASET_REVISION, 110
    )
    assert {name: len(ids) for name, ids in partitions.items()} == {"train": 88, "validation": 11, "test": 11}
    assert stats["action"]["count"] == [32910]
    assert stats["observation.state"]["count"] == [32910]
    assert len(stats["action"]["q99"]) == 54
    manifest = json.loads(manifest_path.read_text())
    assert prepare.partition(manifest["source_episode_ids"]) == partitions


@pytest.mark.parametrize("corruption", ["overlap", "missing", "revision", "stats_hash", "stats_episodes"])
def test_split_loader_rejects_corruption(tmp_path, corruption):
    original = TASK / "splits"
    manifest = json.loads((original / "split_manifest.json").read_text())
    (tmp_path / "train_stats.json").write_bytes((original / "train_stats.json").read_bytes())
    if corruption == "overlap":
        manifest["splits"]["test"][0] = manifest["splits"]["train"][0]
    elif corruption == "missing":
        manifest["splits"]["test"].pop()
    elif corruption == "revision":
        manifest["source"]["revision"] = "another-version"
    elif corruption == "stats_hash":
        (tmp_path / "train_stats.json").write_text("{}")
    else:
        manifest["statistics"]["episodes"] = manifest["splits"]["validation"]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        splits_module.load_episode_splits(path, "fegkkd/bag_v3", prepare.DATASET_REVISION, 110)


def test_held_out_outliers_cannot_affect_training_statistics():
    source_ids = [f"episode-{index}" for index in range(110)]
    partitions = prepare.partition(source_ids)
    vectors = np.repeat(np.arange(110)[:, None], 54, axis=1).astype(float)
    expected = prepare.joint_statistics(vectors[partitions["train"]])
    vectors[partitions["validation"] + partitions["test"]] = 1e9
    assert prepare.joint_statistics(vectors[partitions["train"]]) == expected
    assert prepare.partition(source_ids, seed=43) != partitions


def test_factory_uses_manifest_and_overrides_global_stats_for_both_splits():
    # Exercise the production factory with lightweight dataset/decoder boundaries.
    # Extract only this function so CPU checks do not need optional GPU training imports.
    source = ast.parse((ROOT / "src/lerobot/datasets/factory.py").read_text())
    function = next(
        node
        for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name == "make_train_eval_datasets"
    )
    loaded = []

    class Dataset:
        def __init__(self, *args, episodes=None, **kwargs):
            self.episodes = episodes
            self.num_episodes = 110
            self.meta = SimpleNamespace(stats={"leaked_global_stats": True}, camera_keys=[], depth_keys=[])
            loaded.append(self)

    import copy
    import math

    namespace = {
        "make_dataset": lambda cfg: Dataset(),
        "LeRobotDataset": Dataset,
        "load_episode_splits": splits_module.load_episode_splits,
        "cast_stats_to_numpy": lambda value: value,
        "resolve_delta_timestamps": lambda *args: None,
        "logger": logging.getLogger(__name__),
        "deepcopy": copy.deepcopy,
        "math": math,
    }
    code = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(code), "factory.py", "exec"), namespace)
    data_config = SimpleNamespace(
        eval_split=0,
        split_manifest=str(TASK / "splits/split_manifest.json"),
        repo_id="fegkkd/bag_v3",
        revision=prepare.DATASET_REVISION,
        root="unused",
        image_transforms=SimpleNamespace(enable=False),
        depth_output_unit="m",
        video_backend="pyav",
        repo_type="dataset",
        video_decoder_cache_size=None,
        use_imagenet_stats=False,
    )
    cfg = SimpleNamespace(dataset=data_config, trainable_config=None, rename_map={}, tolerance_s=1e-4)
    train, validation = namespace["make_train_eval_datasets"](cfg)
    manifest = json.loads((TASK / "splits/split_manifest.json").read_text())
    assert train.episodes == manifest["splits"]["train"]
    assert validation.episodes == manifest["splits"]["validation"]
    assert len(loaded) == 3  # no test dataset was constructed
    assert "leaked_global_stats" not in train.meta.stats
    assert train.meta.stats == validation.meta.stats
    assert train.meta.stats is not validation.meta.stats


class Writer:
    def __init__(self):
        self.values = []

    def add_scalar(self, *args):
        self.values.append(args)

    def flush(self):
        pass

    def close(self):
        pass


def test_metrics_append_after_restart_and_write_tensorboard_values(tmp_path):
    writer = Writer()
    for step in (50, 51):
        handler = metrics_module.LocalMetricsHandler(tmp_path, 1, writer)
        tracker = SimpleNamespace(
            steps=step, to_dict=lambda step=step: {"steps": step, "loss": 0.5, "grad_norm": 1.2}
        )
        handler.emit(logging.LogRecord("train", logging.INFO, "", 0, tracker, (), None))
        handler.emit(
            logging.LogRecord("train", logging.INFO, "", 0, f"step {step}: eval_loss=0.4500", (), None)
        )
        handler.close()
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [(row["event"], row["optimizer_updates"]) for row in rows] == [
        ("train", 50),
        ("validation", 50),
        ("train", 51),
        ("validation", 51),
    ]
    assert ("train/grad_norm", 1.2, 51) in writer.values


def test_nonfinite_loss_is_recorded_and_stops_the_run(tmp_path):
    handler = metrics_module.LocalMetricsHandler(tmp_path, 1, Writer())
    tracker = SimpleNamespace(steps=1, to_dict=lambda: {"steps": 1, "loss": float("nan")})
    with pytest.raises(FloatingPointError):
        handler.emit(logging.LogRecord("train", logging.INFO, "", 0, tracker, (), None))
    handler.close()
    assert json.loads((tmp_path / "metrics.jsonl").read_text())["metrics"]["loss"] == "nan"


def test_smoke_config_means_exactly_100_updates_and_frozen_backbone():
    cfg = json.loads((TASK / "train_config.json").read_text())
    assert cfg["steps"] == 100 and cfg["stop_after_steps"] == 50
    assert cfg["accelerator"]["gradient_accumulation"]["steps"] == 1
    assert not cfg["policy"]["tune_llm"] and not cfg["policy"]["tune_visual"]
    assert cfg["policy"]["tune_diffusion_model"] and cfg["policy"]["tune_projector"]
    assert cfg["eval_steps"] == 25 and cfg["max_eval_samples"] == 32
    assert cfg["local_metrics"] and not cfg["wandb"]["enable"]


def test_source_provenance_checksums_are_recorded():
    manifest = json.loads((TASK / "splits/split_manifest.json").read_text())
    stats = (TASK / "splits/train_stats.json").read_bytes()
    assert hashlib.sha256(stats).hexdigest() == manifest["statistics"]["sha256"]
    assert "data/chunk-000/file-000.parquet" in manifest["source"]["files_sha256"]


@pytest.mark.parametrize("restored", [True, False])
def test_checkpoint_acceptance_detects_optimizer_reinitialization(tmp_path, restored):
    from safetensors.numpy import save_file

    verifier = module("verify_run", TASK / "verify_run.py")
    checkpoint = tmp_path / "checkpoints/000100"
    state = checkpoint / "training_state"
    state.mkdir(parents=True)
    (state / "training_step.json").write_text(
        json.dumps({"step": 100, "grad_accum_steps": 1, "dp_world_size": 1})
    )
    (state / "scheduler_state.json").write_text(json.dumps({"last_epoch": 100}))
    (state / "optimizer_param_groups.json").write_text("[]")
    save_file({"rng": np.array([1])}, state / "rng_state.safetensors")
    save_file(
        {"state/0/step": np.array(100 if restored else 50, dtype=np.float32)},
        state / "optimizer_state.safetensors",
    )
    pretrained = checkpoint / "pretrained_model"
    pretrained.mkdir()
    (pretrained / "train_config.json").write_text(json.dumps({"steps": 100}))
    if restored:
        assert verifier.checkpoint_state(tmp_path, 100)["optimizer_step_max"] == 100
    else:
        with pytest.raises(ValueError, match="AdamW"):
            verifier.checkpoint_state(tmp_path, 100)


def test_weight_acceptance_checks_action_update_and_frozen_backbone(tmp_path):
    from safetensors.numpy import save_file

    verifier = module("verify_weights", TASK / "verify_run.py")
    for step in (50, 100):
        directory = tmp_path / "checkpoints" / f"{step:06d}" / "pretrained_model"
        directory.mkdir(parents=True)
        save_file(
            {
                "_groot_model.action_head.vlln.weight": np.array([float(step)]),
                "_groot_model.backbone.norm.weight": np.array([1.0]),
            },
            directory / "model.safetensors",
        )
    assert "vlln.weight" in verifier.compare_weights(tmp_path)["updated_tensor"]
    save_file(
        {
            "_groot_model.action_head.vlln.weight": np.array([100.0]),
            "_groot_model.backbone.norm.weight": np.array([2.0]),
        },
        directory / "model.safetensors",
    )
    with pytest.raises(ValueError, match="frozen"):
        verifier.compare_weights(tmp_path)
