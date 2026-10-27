"""Check the offline runtime without allocating a GPU or loading model weights."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    args = parser.parse_args()
    preparation = args.work_root / "preparation"
    preparation.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    check = subprocess.run(
        ["uv", "pip", "check", "--python", ".venv/bin/python"], capture_output=True, text=True
    )
    output = check.stdout + check.stderr
    known_wheel_metadata_issue = False
    if check.returncode:
        lines = [
            line.strip() for line in output.splitlines() if line.strip() and not line.startswith("Checked ")
        ]
        expected = ["Found 1 incompatibility", "The package `decord` was built for a different platform"]
        if lines != expected or importlib.metadata.version("decord") != "0.6.0":
            raise RuntimeError(output)
        # The locked py3-none wheel's internal WHEEL file says cp36-cp36m.
        # Accept only that exact upstream metadata defect, after a real decode below.
        wheel = importlib.metadata.distribution("decord").read_text("WHEEL")
        if "Tag: cp36-cp36m-manylinux2010_x86_64" not in wheel:
            raise RuntimeError("Unexpected Decord wheel metadata")
        known_wheel_metadata_issue = True

    import av
    import decord
    import torch
    import torchvision
    import transformers
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from torch.utils.tensorboard import SummaryWriter

    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.groot.modeling_groot import GrootPolicy

    config = TrainPipelineConfig.from_pretrained(Path(__file__).parent)
    assert config.steps == 100
    assert GrootPolicy is not None
    with tempfile.TemporaryDirectory(prefix="cpu-probe-", dir=preparation) as temporary:
        folder = Path(temporary)
        video = folder / "probe.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=blue:s=64x64:r=2",
                "-frames:v",
                "2",
                "-pix_fmt",
                "yuv420p",
                "-threads",
                "1",
                str(video),
            ],
            check=True,
        )
        reader = decord.VideoReader(str(video), ctx=decord.cpu(0))
        assert reader.get_batch([0]).asnumpy().shape == (1, 64, 64, 3)
        with av.open(str(video)) as container:
            assert next(container.decode(video=0)).to_ndarray(format="rgb24").shape == (64, 64, 3)
        events = folder / "tensorboard"
        with SummaryWriter(str(events)) as writer:
            writer.add_scalar("preparation/probe", 1.0, 0)
            writer.flush()
        accumulator = EventAccumulator(str(events)).Reload()
        assert accumulator.Scalars("preparation/probe")[0].value == 1.0

    report = {
        "status": "cpu_runtime_validation_passed",
        "gpu_execution_tested": False,
        "typed_config_steps": config.steps,
        "decord_native_decode": True,
        "pyav_native_decode": True,
        "tensorboard_write_read": True,
        "known_upstream_decord_wheel_metadata_issue": known_wheel_metadata_issue,
        "uv_pip_check_output": output,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "transformers": transformers.__version__,
        "uv_lock_sha256": hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest(),
    }
    (preparation / "cpu-runtime-validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
