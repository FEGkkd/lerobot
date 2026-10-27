"""Offline JSONL and TensorBoard logging for ordinary LeRobot training log records."""

import atexit
import datetime as dt
import json
import logging
import math
import re
from pathlib import Path
from typing import Any


class LocalMetricsHandler(logging.Handler):
    def __init__(self, directory: Path, accumulation: int, writer: Any):
        super().__init__(level=logging.INFO)
        directory.mkdir(parents=True, exist_ok=True)
        self.stream = (directory / "metrics.jsonl").open("a", encoding="utf-8")
        self.accumulation = accumulation
        self.writer = writer
        self.closed = False

    def emit(self, record: logging.LogRecord) -> None:
        message = record.msg
        if hasattr(message, "to_dict") and hasattr(message, "steps"):
            values = message.to_dict()
            step = int(values["steps"])
            event = "train"
        else:
            match = re.fullmatch(r"step (\d+): eval_loss=([\deE+\-.naif]+)", record.getMessage())
            if not match:
                return
            step = int(match[1])
            values = {"loss": float(match[2])}
            event = "validation"
        invalid = [
            key
            for key, value in values.items()
            if isinstance(value, (int, float)) and not math.isfinite(value)
        ]
        payload = {
            "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
            "event": event,
            "micro_step": step,
            "optimizer_updates": step // self.accumulation,
            "metrics": {key: str(value) if key in invalid else value for key, value in values.items()},
        }
        self.stream.write(json.dumps(payload, allow_nan=False) + "\n")
        self.stream.flush()
        if invalid:
            raise FloatingPointError(f"Non-finite training metrics: {invalid}")
        for key, value in values.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"{event}/{key}", value, step // self.accumulation)
        self.writer.flush()

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.stream.close()
            self.writer.close()
        super().close()


def configure_local_metrics(output_dir: Path, accumulation: int) -> LocalMetricsHandler:
    # This dependency is required only when explicitly enabling local_metrics.
    from torch.utils.tensorboard import SummaryWriter

    directory = Path(output_dir) / "logs"
    writer = SummaryWriter(log_dir=str(directory / "tensorboard"))
    handler = LocalMetricsHandler(directory, accumulation, writer)
    logging.getLogger().addHandler(handler)
    atexit.register(handler.close)
    return handler


def write_parameter_report(policy: Any, output_dir: Path) -> None:
    groups: dict[str, dict[str, int]] = {}
    for name, parameter in policy.named_parameters():
        group = "backbone" if ".backbone." in name else "action_head" if ".action_head." in name else "other"
        counts = groups.setdefault(group, {"trainable": 0, "frozen": 0})
        counts["trainable" if parameter.requires_grad else "frozen"] += parameter.numel()
    target = Path(output_dir) / "logs" / "trainable_parameters.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(groups, indent=2) + "\n", encoding="utf-8")
    logging.info("Parameter groups: %s", groups)
