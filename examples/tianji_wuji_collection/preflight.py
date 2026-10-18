#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Observe collector health and low-dimensional feedback. Never command hardware."""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from collection_tools import (
    TASK_DIR,
    finite_number,
    load_document,
    ordered_positions,
    save_json,
    storage_budget,
    validate_site,
    validate_task,
)


def assess_snapshot(task: dict[str, Any], snapshot: dict[str, Any], active: bool) -> dict[str, Any]:
    errors = []
    limits = task["preflight"]
    collector = snapshot.get("collector", {})
    ages = snapshot.get("arrival_age_ms", {})
    if ages.get("collector", float("inf")) > limits["max_callback_age_ms"]:
        errors.append("collector status heartbeat is absent or stale")
    if collector.get("state") != "IDLE":
        errors.append("collector must be IDLE before starting a new pilot recording")
    if not collector.get("collector_instance_id") or collector.get("error"):
        errors.append("collector instance identity missing or collector error present")
    if snapshot.get("post_warmup_status_count", 0) < 2:
        errors.append("insufficient collector status observations after warmup")
    status = {s["stream_id"]: s for s in collector.get("streams", [])}
    for stream in task["streams"]:
        stream_id = stream["id"]
        source = status.get(stream_id, {})
        if not source.get("required") or not source.get("enabled"):
            errors.append(f"{stream_id}: collector is not using the required task profile")
        if source.get("topic") != stream["topic"] or source.get("message_type") != stream["type"]:
            errors.append(f"{stream_id}: collector topic/type mismatch")
        if source.get("error"):
            errors.append(f"{stream_id}: {source['error']}")
        publishers = snapshot.get("publishers", {}).get(stream["topic"], [])
        if len(publishers) != 1:
            errors.append(f"{stream_id}: expected one publisher endpoint, found {len(publishers)}")
        needs_sample = stream["role"] != "action" or active
        if needs_sample:
            age = source.get("last_sample_age_ms", -1)
            if (
                not source.get("received")
                or not finite_number(age)
                or age < 0
                or age > limits["max_callback_age_ms"]
            ):
                errors.append(f"{stream_id}: samples are absent or stale at collector")
        if stream["role"] in ("image", "state"):
            minimum = limits["min_camera_hz"] if stream["role"] == "image" else limits["min_joint_hz"]
            measured = snapshot.get("minimum_receive_hz", {}).get(stream_id, 0)
            if not finite_number(measured) or measured < minimum:
                errors.append(f"{stream_id}: observed minimum {measured:.1f} Hz < {minimum} Hz")
        if stream["role"] == "image":
            baseline = snapshot.get("baseline_source_gaps", {}).get(stream_id)
            if source.get("source_sequence_available") and (
                baseline is None or source.get("source_sequence_gaps", 0) > baseline
            ):
                errors.append(f"{stream_id}: source camera sequence gaps occurred during observation")
        if stream["role"] == "state" or (active and stream["role"] == "action"):
            message = snapshot.get("joint_messages", {}).get(stream_id)
            if message is None or ages.get(stream_id, float("inf")) > limits["max_callback_age_ms"]:
                errors.append(f"{stream_id}: no fresh low-dimensional observation")
                continue
            try:
                ordered_positions(message, stream, task["alignment"]["max_hand_feedback_age_ms"])
            except ValueError as exc:
                errors.append(f"{stream_id}: {exc}")
            if (
                active
                and stream_id.startswith("wuji")
                and stream["role"] == "state"
                and message.get("enabled") is not True
            ):
                errors.append(f"{stream_id}: hand is not enabled in active phase")
            if active and stream_id == "tianji_command":
                deadline = message.get("valid_until", {})
                deadline_ns = deadline.get("sec", 0) * 10**9 + deadline.get("nanosec", 0)
                if not message.get("session_id") or deadline_ns <= snapshot.get("ros_now_ns", 0):
                    errors.append("tianji_command: empty session or expired command")
    robot = snapshot.get("robot_state", {})
    if ages.get("robot_state", float("inf")) > limits["max_callback_age_ms"]:
        errors.append("robot state heartbeat is absent or stale")
    if not robot.get("connected") or not robot.get("homed") or not robot.get("feedback_valid"):
        errors.append("robot must report connected, homed and valid feedback")
    if (
        robot.get("left_error")
        or robot.get("right_error")
        or robot.get("hardware_state_name")
        in (
            "HOLD",
            "ERROR",
            "HOMING",
            "DISCONNECTED",
        )
    ):
        errors.append("robot reports an error or an unsuitable lifecycle state")
    if active and not all(robot.get(key) for key in ("enabled", "command_allowed", "motion_allowed")):
        errors.append("active phase requires enabled and commandable robot feedback")
    return {
        "data_checks_passed": not errors,
        "errors": errors,
        "phase": "active" if active else "connected",
        "scope": "Read-only data checks, not hardware motion/safety certification or sample alignment proof.",
    }


def observe(task: dict[str, Any], seconds: float, warmup: float) -> dict[str, Any]:
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from rosidl_runtime_py.convert import message_to_ordereddict
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:
        raise ValueError(
            "Source ROS 2 Jazzy, teleop and platform interface overlays before live preflight"
        ) from exc

    result: dict[str, Any] = {
        "joint_messages": {},
        "minimum_receive_hz": {},
        "baseline_source_gaps": {},
        "post_warmup_status_count": 0,
    }
    received: dict[str, float] = {}
    started = time.monotonic()
    rclpy.init()
    node = Node("apple_collection_preflight")
    subscriptions = []

    def receive(key: str, message: Any) -> None:
        received[key] = time.monotonic()
        value = dict(message_to_ordereddict(message))
        if key == "collector":
            result[key] = value
            if received[key] - started < warmup:
                return
            result["post_warmup_status_count"] += 1
            for source in value["streams"]:
                stream_id = source["stream_id"]
                previous = result["minimum_receive_hz"].get(stream_id, float("inf"))
                result["minimum_receive_hz"][stream_id] = min(previous, source["receive_hz"])
                result["baseline_source_gaps"].setdefault(stream_id, source["source_sequence_gaps"])
        elif key == "robot_state":
            result[key] = value
        else:
            result["joint_messages"][key] = value

    try:
        status_qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        feedback_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        topics = [
            (
                "collector",
                "/dex/collection/status",
                "dex_collection_interfaces/msg/CollectionStatus",
                status_qos,
            ),
            ("robot_state", "/tianji/robot_state", "tianji_interfaces/msg/RobotState", status_qos),
        ]
        topics.extend(
            (s["id"], s["topic"], s["type"], feedback_qos) for s in task["streams"] if s["role"] != "image"
        )
        for key, topic, message_type, qos in topics:
            subscriptions.append(
                node.create_subscription(
                    get_message(message_type), topic, lambda message, name=key: receive(name, message), qos
                )
            )
        while rclpy.ok() and time.monotonic() - started < seconds:
            rclpy.spin_once(node, timeout_sec=0.1)
        result["publishers"] = {}
        for stream in task["streams"]:
            result["publishers"][stream["topic"]] = [
                {"node_name": p.node_name, "namespace": p.node_namespace, "topic_type": p.topic_type}
                for p in node.get_publishers_info_by_topic(stream["topic"])
            ]
        now = time.monotonic()
        result["arrival_age_ms"] = {key: (now - stamp) * 1000 for key, stamp in received.items()}
        result["ros_now_ns"] = node.get_clock().now().nanoseconds
        result["observation_duration_sec"] = now - started
        result["use_sim_time"] = node.get_parameter("use_sim_time").value
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, default=TASK_DIR / "task.json")
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--phase", choices=("connected", "active"), default="connected")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        task, site = load_document(args.task), load_document(args.site)
        errors = validate_task(task) + validate_site(task, site)
        if errors:
            raise ValueError("; ".join(errors))
        if args.seconds < 5 or args.seconds > 60:
            raise ValueError("observation duration must be in [5, 60] seconds")
        actual_domain = int(os.environ.get("ROS_DOMAIN_ID", "0"))
        if actual_domain != site["ros_domain_id"]:
            raise ValueError(f"ROS_DOMAIN_ID={actual_domain} differs from measured site configuration")
        root = Path(site["data_root"])
        if not root.is_dir():
            raise ValueError("recording data_root must already exist on this host")
        budget = storage_budget(task, site, args.trials)
        free = shutil.disk_usage(root).free
        if free < budget["required_free_bytes"]:
            raise ValueError(f"disk free {free} < planned-session budget {budget['required_free_bytes']}")
        snapshot = observe(task, args.seconds, task["preflight"]["warmup_seconds"])
        result = assess_snapshot(task, snapshot, args.phase == "active")
        if snapshot["use_sim_time"]:
            result["errors"].append("real collection preflight must not use simulated ROS time")
            result["data_checks_passed"] = False
        result.update(
            {
                "ros_domain_id": actual_domain,
                "free_bytes": free,
                "storage_budget": budget,
                "snapshot": snapshot,
            }
        )
        save_json(args.out, result)
        print(json.dumps({key: value for key, value in result.items() if key != "snapshot"}, indent=2))
        return 0 if result["data_checks_passed"] else 1
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
