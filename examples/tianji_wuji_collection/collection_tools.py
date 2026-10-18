#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Prepare an apple-to-bowl collection session without controlling any device."""

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

TASK_DIR = Path(__file__).resolve().parents[2] / "tasks" / "apple_to_blue_bowl_v1"


def load_document(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise ValueError("YAML input requires PyYAML: uv run --with pyyaml python ...") from exc
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Refuse to overwrite previous operator records or configuration snapshots.
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def finite_number(value: Any) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def finite_vector(value: Any, length: int) -> bool:
    return isinstance(value, list) and len(value) == length and all(finite_number(x) for x in value)


def validate_task(task: dict[str, Any]) -> list[str]:
    errors = []
    dataset = task.get("dataset", {})
    names = dataset.get("joint_names", [])
    if len(names) != 54 or len(set(names)) != 54:
        errors.append("dataset.joint_names must contain 54 unique names")
    if dataset.get("state_dim") != 54 or dataset.get("action_dim") != 54:
        errors.append("state/action dimensions must both be 54")
    streams = task.get("streams", [])
    if len(streams) != 10 or len({s["id"] for s in streams}) != 10:
        errors.append("exactly 10 unique required streams are expected")
    for role in ("state", "action"):
        flattened = [name for stream in streams if stream["role"] == role for name in stream["joint_names"]]
        if flattened != names:
            errors.append(f"{role} joint order disagrees with dataset.joint_names")
    if task.get("allowed_grasp_hands") != ["left", "right"]:
        errors.append("this task must permit both left and right grasps")
    if dataset.get("fps") != 30 or task.get("head_fixed") is not True:
        errors.append("this task contract requires 30 Hz export and a fixed head")
    return errors


def validate_site(task: dict[str, Any], site: dict[str, Any]) -> list[str]:
    errors = []
    if site.get("task_id") != task["task_id"]:
        errors.append("site.task_id does not match task")
    for key in ("operator_name", "data_root"):
        if not isinstance(site.get(key), str) or not site[key].strip():
            errors.append(f"site.{key} must be filled")
    root = site.get("data_root") or ""
    if not (PurePosixPath(root).is_absolute() or PureWindowsPath(root).is_absolute()):
        errors.append("site.data_root must be an absolute recording-host path")
    domain = site.get("ros_domain_id")
    if type(domain) is not int or not 0 <= domain <= 232:
        errors.append("site.ros_domain_id must be an explicit integer in [0, 232]")
    for section, key in (
        ("apple", "diameter_m"),
        ("bowl", "inner_diameter_m"),
        ("bowl", "inner_depth_m"),
        ("table", "height_m"),
    ):
        value = site.get(section, {}).get(key)
        if not finite_number(value) or value <= 0:
            errors.append(f"site.{section}.{key} must be measured and positive")
    if not site.get("apple", {}).get("material_or_type"):
        errors.append("site.apple.material_or_type must describe the actual apple")
    if not site.get("table", {}).get("frame_description"):
        errors.append("site.table.frame_description must define the marked table origin and axes")
    table = site.get("table", {})
    for key in ("apple_x_bounds_m", "apple_y_bounds_m", "bowl_center_xy_m"):
        value = table.get(key)
        if not finite_vector(value, 2):
            errors.append(f"site.table.{key} must contain two measured finite values")
        elif "bounds" in key and value[0] >= value[1]:
            errors.append(f"site.table.{key} requires lower < upper")
    for key, length in (("left_arm", 7), ("right_arm", 7), ("left_hand", 20), ("right_hand", 20)):
        if not finite_vector(site.get("prepare_pose_rad", {}).get(key), length):
            errors.append(f"site.prepare_pose_rad.{key} requires {length} measured joint positions")
    for camera in ("front", "left_wrist", "right_wrist"):
        values = site.get("cameras", {}).get(camera, {})
        for key in ("width", "height", "fps", "channels"):
            if type(values.get(key)) is not int or values[key] <= 0:
                errors.append(f"site.cameras.{camera}.{key} must be a positive integer")
        if values.get("channels") != 3:
            errors.append(f"site.cameras.{camera} must describe a 3-channel color stream")
        if values.get("fps", 0) < task["dataset"]["fps"]:
            errors.append(f"site.cameras.{camera}.fps is below the export frequency")
    confirmations = (
        "camera_geometry_and_color",
        "head_fixed",
        "joint_order_and_radian_units",
        "real_driver_configs",
        "prepare_pose_reachable",
        "apple_and_bowl_visible",
        "storage_root_on_recording_host",
    )
    for key in confirmations:
        if site.get("verified", {}).get(key) is not True:
            errors.append(f"site.verified.{key} requires an actual on-site check")
    for repository in ("lerobot", "dex_data_platform", "whole_body_teleop"):
        revision = site.get("source_revisions", {}).get(repository, "")
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            errors.append(f"site.source_revisions.{repository} requires the actual deployed commit SHA")
    return errors


def storage_budget(task: dict[str, Any], site: dict[str, Any], trials: int) -> dict[str, int]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    rate = sum(c["width"] * c["height"] * c["channels"] * c["fps"] for c in site["cameras"].values())
    raw = rate * task["max_duration_sec"] * trials
    required = math.ceil(raw * task["preflight"]["storage_margin"]) + task["preflight"]["minimum_free_bytes"]
    return {"raw_rgb_bytes_per_sec": rate, "raw_rgb_bytes": raw, "required_free_bytes": required}


def prepare_profile(task: dict[str, Any], profile: dict[str, Any], site: dict[str, Any]) -> dict[str, Any]:
    # JSON is also valid YAML for both yaml-cpp and the platform gateway's YAML reader.
    result = json.loads(json.dumps(profile))
    required = {stream["id"]: stream for stream in task["streams"]}
    auxiliary = set(task["auxiliary_stream_ids"])
    for stream_id, source in result["streams"].items():
        source["selected"] = stream_id in required or stream_id in auxiliary
        source["required"] = stream_id in required
    for stream_id, contract in required.items():
        source = result["streams"].get(stream_id)
        if source is None or source["topic"] != contract["topic"] or source["type"] != contract["type"]:
            raise ValueError(f"Profile schema/topic mismatch for {stream_id}")
        if not source.get("enabled"):
            raise ValueError(f"Required stream {stream_id} is disabled")
    result["collection"]["data_root"] = site["data_root"]
    result["collection"]["minimum_free_bytes"] = task["preflight"]["minimum_free_bytes"]
    return result


def ordered_positions(
    message: dict[str, Any], stream: dict[str, Any], max_hand_age_ms: float = 100
) -> list[float]:
    names = message.get("joint_names", message.get("name", []))
    values = message.get(stream["position_field"], [])
    if len(names) != len(values) or len(set(names)) != len(names):
        raise ValueError("invalid joint name/value cardinality or duplicate names")
    mapping = dict(zip(names, values, strict=True))
    if set(names) != set(stream["joint_names"]):
        raise ValueError("joint names do not match the task contract")
    ordered = [mapping[name] for name in stream["joint_names"]]
    if not finite_vector(ordered, len(stream["joint_names"])):
        raise ValueError("joint positions contain non-finite or non-numeric values")
    if stream["id"].startswith("wuji") and stream["role"] == "state":
        if message.get("connected") is not True:
            raise ValueError("hand is disconnected")
        valid = message.get("position_valid", [])
        online = message.get("online", [])
        stale = message.get("stale", [])
        ages = message.get("age_ms", [])
        if len(valid) != 20 or not all(v is True for v in valid):
            raise ValueError("hand position_valid must be true for all 20 joints")
        if len(online) != 20 or not all(v is True for v in online):
            raise ValueError("all 20 hand joints must be online")
        if len(stale) != 20 or any(v is not False for v in stale):
            raise ValueError("hand has stale or missing joint flags")
        if not finite_vector(ages, 20) or any(age < 0 or age > max_hand_age_ms for age in ages):
            raise ValueError(f"hand SDK sample age exceeds the {max_hand_age_ms} ms data-quality budget")
        source = message.get("feedback_source", "").lower()
        if source != "wuji_hand2":
            raise ValueError("hand feedback_source must be wuji_hand2 for this real-data contract")
    return ordered


def audit_manifest(task: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    errors, warnings = [], []
    if manifest.get("format") != "dex-native-v2" or manifest.get("format_version") != 2:
        errors.append("unsupported raw manifest format")
    if manifest.get("state") != "COMPLETED" or manifest.get("error"):
        errors.append("episode did not finalize successfully")
    if not isinstance(manifest.get("episode_id"), str) or not manifest["episode_id"]:
        errors.append("episode identity is missing")
    start, stop = manifest.get("start_monotonic_ns"), manifest.get("stop_monotonic_ns")
    if type(start) is not int or type(stop) is not int or not 0 < start < stop:
        errors.append("episode admission interval is invalid")
    if manifest.get("runtime", {}).get("source_mode") != "external":
        errors.append("real demonstration requires external source_mode")
    if manifest.get("metadata", {}).get("task_name") != task["task_id"]:
        errors.append("manifest metadata.task_name does not match the task")
    if manifest.get("quality_state") != "good":
        warnings.append("global episode quality is not good; inspect optional as well as required streams")
    selected = set(manifest.get("selected_streams", []))
    skipped = manifest.get("skipped_streams", {})
    for stream in task["streams"]:
        stream_id = stream["id"]
        info = manifest.get("streams", {}).get(stream_id, {})
        if stream_id not in selected or stream_id in skipped or info.get("recorded") is not True:
            errors.append(f"{stream_id}: missing/skipped/not recorded")
            continue
        if info.get("topic") != stream["topic"] or info.get("type") != stream["type"]:
            errors.append(f"{stream_id}: topic/type mismatch")
        if not info.get("written", 0) or info.get("accepted") != info.get("written"):
            errors.append(f"{stream_id}: zero samples or accepted/written mismatch")
        counters = ("dropped", "write_errors", "unwritten", "queued_samples", "write_in_flight")
        if any(type(info.get(key)) is not int or info[key] != 0 for key in counters) or info.get("error"):
            errors.append(f"{stream_id}: missing counters, queue drops, write errors or unfinished writes")
        if info.get("source_sequence_available") and info.get("episode_source_sequence_gaps", 0):
            errors.append(f"{stream_id}: upstream camera sequence gaps")
    return {
        "manifest_checks_passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "scope": "Manifest only; sample timing, payload validity and task success require separate review.",
    }


def make_annotation(task: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task_id": task["task_id"],
        "episode_id": manifest["episode_id"],
        "manifest_sha256": digest(manifest),
        "result": None,
        "failure_reason": None,
        "grasp_hand": None,
        "apple_zone": None,
        "bowl_zone": None,
        "segment_start_offset_sec": None,
        "segment_end_offset_sec": None,
        "stable_in_bowl_sec": None,
        "fully_released": None,
        "human_intervention": None,
        "reviewer": None,
        "notes": "Offsets are relative to manifest.start_monotonic_ns, not ROS time.",
    }


def validate_annotation(
    task: dict[str, Any], manifest: dict[str, Any], annotation: dict[str, Any]
) -> list[str]:
    errors = []
    if annotation.get("task_id") != task["task_id"] or annotation.get("episode_id") != manifest.get(
        "episode_id"
    ):
        errors.append("annotation task/episode identity mismatch")
    if annotation.get("manifest_sha256") != digest(manifest):
        errors.append("annotation manifest digest does not match")
    result = annotation.get("result")
    if result not in ("success", "failure", "aborted"):
        errors.append("result must be success, failure or aborted")
    if annotation.get("grasp_hand") not in task["allowed_grasp_hands"]:
        errors.append("grasp_hand must record the actual left or right hand")
    for key in ("apple_zone", "bowl_zone"):
        if annotation.get(key) not in task["collection_plan"]["position_zones"]:
            errors.append(f"{key} must be left, center or right in the marked table frame")
    start, end = annotation.get("segment_start_offset_sec"), annotation.get("segment_end_offset_sec")
    duration = (manifest.get("stop_monotonic_ns", 0) - manifest.get("start_monotonic_ns", 0)) / 1e9
    if not finite_number(start) or not finite_number(end) or not 0 <= start < end <= duration:
        errors.append("task segment must be a nonempty interval within the raw episode")
    elif end - start > task["max_duration_sec"]:
        errors.append("task segment exceeds the task timeout")
    if not annotation.get("reviewer"):
        errors.append("reviewer must be filled after video/payload review")
    if result == "success":
        dwell = annotation.get("stable_in_bowl_sec")
        if not finite_number(dwell) or dwell < task["success"]["stable_seconds"]:
            errors.append("success requires measured stable dwell in the bowl")
        elif finite_number(start) and finite_number(end) and dwell > end - start:
            errors.append("stable dwell cannot exceed the task segment duration")
        if annotation.get("fully_released") is not True or annotation.get("human_intervention") is not False:
            errors.append("success requires full release and no human intervention")
        if annotation.get("failure_reason") is not None:
            errors.append("successful episode cannot have a failure reason")
    elif result in ("failure", "aborted") and annotation.get("failure_reason") not in task["failure_reasons"]:
        errors.append("failed/aborted episode requires a failure reason")
    return errors


def stratified_split(records: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    groups: dict[str, list[str]] = {"left": [], "right": []}
    for record in records:
        if record.get("result") != "success" or record.get("grasp_hand") not in groups:
            raise ValueError("split input must contain only reviewed successful left/right episodes")
        if not record.get("reviewer") or not record.get("manifest_sha256"):
            raise ValueError("split input requires reviewer and manifest identity from completed annotations")
        episode = record.get("episode_id")
        if not isinstance(episode, str) or not episode:
            raise ValueError("split input contains an empty episode ID")
        groups[record["grasp_hand"]].append(episode)
    ids = groups["left"] + groups["right"]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate episode ID in split input")
    if any(len(group) < 2 for group in groups.values()):
        raise ValueError("need at least two successful episodes per hand for a train/validation split")
    rng, train, validation = random.Random(seed), [], []
    for group in groups.values():
        group.sort()
        rng.shuffle(group)
        count = max(1, math.ceil(len(group) * 0.1))
        validation.extend(group[:count])
        train.extend(group[count:])
    return {
        "seed": seed,
        "train_episode_ids": sorted(train),
        "validation_episode_ids": sorted(validation),
        "note": "Map raw IDs to exported episode indices; preserve this explicit split, not the last 10%.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, default=TASK_DIR / "task.json")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="Check task and optional measured site configuration")
    validate.add_argument("--site", type=Path)
    session = sub.add_parser("session", help="Write a session snapshot/profile/blank trial log; no ROS calls")
    session.add_argument("--site", type=Path, required=True)
    session.add_argument("--profile", type=Path, default=TASK_DIR / "collection.yaml")
    session.add_argument("--out", type=Path, required=True)
    session.add_argument("--trials", type=int, default=5)
    session.add_argument("--session-name", required=True)
    audit = sub.add_parser("audit", help="Inspect a finalized manifest and optional human annotation")
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--annotation", type=Path)
    audit.add_argument("--out", type=Path)
    annotate = sub.add_parser("annotation-template", help="Create an unfilled post-recording annotation")
    annotate.add_argument("--manifest", type=Path, required=True)
    annotate.add_argument("--out", type=Path, required=True)
    split = sub.add_parser("split", help="Freeze an episode split stratified by actual grasp hand")
    split.add_argument("--annotations", type=Path, nargs="+", required=True)
    split.add_argument("--seed", type=int, default=1000)
    split.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        task = load_document(args.task)
        errors = validate_task(task)
        if errors:
            raise ValueError("; ".join(errors))
        if args.command == "validate":
            errors = validate_site(task, load_document(args.site)) if args.site else []
            print(
                json.dumps({"task_valid": True, "site_checked": bool(args.site), "errors": errors}, indent=2)
            )
            return 1 if errors else 0
        if args.command == "session":
            site = load_document(args.site)
            errors = validate_site(task, site)
            if errors:
                raise ValueError("; ".join(errors))
            budget = storage_budget(task, site, args.trials)
            profile = prepare_profile(task, load_document(args.profile), site)
            args.out.mkdir(parents=True, exist_ok=False)
            save_json(args.out / "task.snapshot.json", task)
            save_json(args.out / "site.snapshot.json", site)
            save_json(args.out / "collection.generated.yaml", profile)
            metadata = {
                "task_name": task["task_id"],
                "operator_name": site["operator_name"],
                "object_name": f"{site['apple']['id']} + {site['bowl']['id']}",
                "session_name": args.session_name,
                "has_trial_index": True,
                "trial_index": 1,
                "notes": f"task_sha256={digest(task)}; site_sha256={digest(site)}",
            }
            save_json(
                args.out / "session.json",
                {
                    "created_utc": datetime.now(UTC).isoformat(),
                    "task_sha256": digest(task),
                    "site_sha256": digest(site),
                    "planned_trials": args.trials,
                    "storage_budget": budget,
                    "platform_metadata_example": metadata,
                    "instruction": "Enter metadata in existing platform UI. Increment trial_index per attempt.",
                },
            )
            with (args.out / "trials.csv").open("x", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "trial_index",
                        "episode_id",
                        "apple_zone",
                        "bowl_zone",
                        "grasp_hand",
                        "result",
                        "segment_start_offset_sec",
                        "segment_end_offset_sec",
                        "notes",
                    ]
                )
                writer.writerows([[i, "", "", "", "", "", "", "", ""] for i in range(1, args.trials + 1)])
            print(json.dumps({"session_directory": str(args.out), "storage_budget": budget}, indent=2))
        elif args.command == "annotation-template":
            save_json(args.out, make_annotation(task, load_document(args.manifest)))
        elif args.command == "audit":
            manifest = load_document(args.manifest)
            result = audit_manifest(task, manifest)
            if args.annotation:
                result["errors"].extend(validate_annotation(task, manifest, load_document(args.annotation)))
            result["checks_passed"] = not result["errors"]
            result["annotation_checked"] = bool(args.annotation)
            if args.out:
                save_json(args.out, result)
            print(json.dumps(result, indent=2))
            return 0 if result["checks_passed"] else 1
        elif args.command == "split":
            save_json(args.out, stratified_split([load_document(p) for p in args.annotations], args.seed))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
