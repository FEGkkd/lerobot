"""Exercise real Adam/RNG checkpoint continuation and the moving-loss gate on CPU."""

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from lerobot.optim.schedulers import ContinuationConstantSchedulerConfig  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "continue_recipe", Path(__file__).parents[1] / "continue_training.py"
)
recipe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recipe)


def test_real_adam_schedule_restore_and_following_process(tmp_path):
    """Changing LR must retain the actual moment tensors and global step on restart."""
    p = torch.nn.Parameter(torch.tensor([1.0]))
    opt = torch.optim.AdamW([p], lr=5e-5)
    p.grad = torch.ones_like(p)
    opt.step()
    opt.state[p]["step"].fill_(30000)
    old = opt.state_dict()
    expected_moment = old["state"][0]["exp_avg"].clone()
    peak = 1e-5
    for g in old["param_groups"]:
        g.update(lr=peak / 500, initial_lr=peak)
    q = torch.nn.Parameter(p.detach().clone())
    resumed = torch.optim.AdamW([q], lr=peak)
    sch = ContinuationConstantSchedulerConfig().build(resumed, 300000)
    resumed.load_state_dict(old)
    state = sch.state_dict()
    state.update(last_epoch=30000, _step_count=30001, base_lrs=[peak], _last_lr=[peak / 500])
    sch.load_state_dict(state)
    assert torch.equal(resumed.state[q]["exp_avg"], expected_moment)
    for _ in range(500):
        q.grad = torch.ones_like(q)
        resumed.step()
        sch.step()
    assert resumed.state[q]["step"].item() == 30500
    assert sch.last_epoch == 30500
    assert resumed.param_groups[0]["lr"] == pytest.approx(peak)
    next_p = torch.nn.Parameter(q.detach().clone())
    next_opt = torch.optim.AdamW([next_p], lr=peak)
    next_sch = ContinuationConstantSchedulerConfig().build(next_opt, 300000)
    next_opt.load_state_dict(resumed.state_dict())
    next_sch.load_state_dict(sch.state_dict())
    next_p.grad = torch.ones_like(next_p)
    next_opt.step()
    next_sch.step()
    assert next_opt.state[next_p]["step"].item() == 30501
    assert next_sch.last_epoch == 30501
    assert next_opt.param_groups[0]["lr"] == pytest.approx(peak)


def test_staging_does_not_edit_source_and_keeps_adam_reference(tmp_path):
    source = tmp_path / "source"
    cp = source / "train/checkpoints/030000"
    (cp / "pretrained_model").mkdir(parents=True)
    (cp / "training_state").mkdir()
    recipe.write(source / "verification.json", {"status": "passed", "optimizer_updates": 30000})
    recipe.write(
        cp / "training_state/training_step.json",
        {
            "step": 30000,
            "batch_size": 4,
            "grad_accum_steps": 1,
            "dp_world_size": 1,
        },
    )
    recipe.write(cp / "training_state/scheduler_state.json", {"last_epoch": 30000, "base_lrs": [5e-5]})
    recipe.write(cp / "training_state/optimizer_param_groups.json", [{"params": [0], "lr": 0.0}])
    recipe.write(
        cp / "pretrained_model/train_config.json",
        {
            "optimizer": {"type": "adamw", "lr": 5e-5},
            "policy": {},
            "batch_size": 4,
            "accelerator": {"gradient_accumulation": {"steps": 1}},
        },
    )
    for file in [
        cp / "pretrained_model/model.safetensors",
        cp / "training_state/optimizer_state.safetensors",
        cp / "training_state/rng_state.safetensors",
    ]:
        file.write_bytes(b"immutable-fixture")
    before = {str(x): x.read_bytes() for x in cp.rglob("*") if x.is_file()}
    job = tmp_path / "new_job"
    job.mkdir()
    config = recipe.stage_resume(source, job)
    assert recipe.read(config)["steps"] == 300000
    assert recipe.read(config)["output_dir"] == str(job / "train")
    staged = config.parents[1] / "training_state/optimizer_state.safetensors"
    assert staged.is_symlink()
    assert staged.resolve() == (cp / "training_state/optimizer_state.safetensors").resolve()
    assert before == {str(x): x.read_bytes() for x in cp.rglob("*") if x.is_file()}
    with pytest.raises(FileExistsError):
        recipe.stage_resume(source, job)


@pytest.mark.parametrize("step,expected", [(29999, 0), (30000, 0.002), (30499, 1), (300000, 1)])
def test_global_warmup_boundary(step, expected):
    assert ContinuationConstantSchedulerConfig().factor(step) == pytest.approx(expected)


def test_moving_window_is_not_a_single_low_value():
    window = [{"metrics": {"loss": x}} for x in [0.01] * 99 + [0.0001]]
    assert recipe.moving_loss(window) > recipe.TARGET
    assert recipe.moving_loss([{"metrics": {"loss": 0.0009}}] * 100) <= recipe.TARGET
    with pytest.raises(ValueError):
        recipe.moving_loss(window[:50])


def test_phase_verifier_rejects_a_missing_update(tmp_path):
    import numpy as np
    from safetensors.numpy import save_file
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    job = tmp_path / "job"
    run = job / "train"
    logs = run / "logs"
    (logs / "tensorboard").mkdir(parents=True)
    (job / "logs").mkdir()
    (job / "logs/phase-035000.log").write_text("Resuming data order at epoch 3, sample 21270")
    recipe.write(
        logs / "trainable_parameters.json",
        {
            "backbone": {"trainable": 0},
            "action_head": {"trainable": 1620515968},
        },
    )
    cp = run / "checkpoints/035000"
    (cp / "training_state").mkdir(parents=True)
    (cp / "pretrained_model").mkdir()
    recipe.write(
        cp / "training_state/training_step.json",
        {
            "step": 35000,
            "batch_size": 4,
            "grad_accum_steps": 1,
        },
    )
    recipe.write(cp / "training_state/scheduler_state.json", {"last_epoch": 35000, "_last_lr": [1e-5]})
    recipe.write(
        cp / "pretrained_model/train_config.json",
        {
            "steps": 300000,
            "scheduler": {"type": "continuation_constant"},
        },
    )
    save_file({"state/0/step": np.array(35000.0)}, cp / "training_state/optimizer_state.safetensors")
    rows = []
    writer = EventFileWriter(str(logs / "tensorboard"))
    for step in range(30010, 35001, 10):
        metrics = {"loss": 0.0009, "gpu_mem_gb": 36.1, "lr": 1e-5, "samples": 4 * step}
        rows.append({"event": "train", "optimizer_updates": step, "metrics": metrics})
        writer.add_event(
            Event(
                step=step,
                summary=Summary(
                    value=[
                        Summary.Value(tag="train/" + key, simple_value=value)
                        for key, value in metrics.items()
                    ]
                ),
            )
        )
        if step % 1000 == 0:
            rows.append({"event": "validation", "optimizer_updates": step, "metrics": {"loss": 0.1}})
            writer.add_event(
                Event(
                    step=step,
                    summary=Summary(
                        value=[
                            Summary.Value(tag="validation/loss", simple_value=0.1),
                        ]
                    ),
                )
            )
    writer.close()
    import json

    path = logs / "metrics.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in rows))
    result = recipe.verify_phase(job, 35000)
    assert result["target_reached"] and result["optimizer_step"] == 35000
    path.write_text("\n".join(json.dumps(x) for x in rows[1:]))
    with pytest.raises(ValueError, match="Missing/repeated continuation"):
        recipe.verify_phase(job, 35000)
