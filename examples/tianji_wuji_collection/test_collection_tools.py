#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Offline positive/negative controls; no ROS graph or hardware is accessed."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from collection_tools import (
    TASK_DIR,
    audit_manifest,
    load_document,
    main,
    make_annotation,
    ordered_positions,
    prepare_profile,
    save_json,
    storage_budget,
    stratified_split,
    validate_annotation,
    validate_site,
    validate_task,
)
from preflight import assess_snapshot

TASK = load_document(TASK_DIR / "task.json")


def synthetic_site():
    site = load_document(TASK_DIR / "site.example.json")
    site.update(operator_name="SYNTHETIC_TEST_ONLY", data_root="/synthetic/data", ros_domain_id=197)
    site["apple"].update(material_or_type="synthetic fixture", diameter_m=0.07)
    site["bowl"].update(inner_diameter_m=0.2, inner_depth_m=0.08)
    site["table"].update(
        frame_description="synthetic tabletop",
        height_m=0.75,
        apple_x_bounds_m=[-0.2, 0.2],
        apple_y_bounds_m=[0.2, 0.3],
        bowl_center_xy_m=[0, 0.4],
    )
    site["prepare_pose_rad"] = {
        "left_arm": [0.0] * 7,
        "right_arm": [0.0] * 7,
        "left_hand": [0.0] * 20,
        "right_hand": [0.0] * 20,
    }
    site["verified"] = dict.fromkeys(site["verified"], True)
    return site


def synthetic_manifest():
    streams = {}
    for stream in TASK["streams"]:
        streams[stream["id"]] = {
            "recorded": True,
            "topic": stream["topic"],
            "type": stream["type"],
            "written": 300,
            "accepted": 300,
            "dropped": 0,
            "write_errors": 0,
            "unwritten": 0,
            "queued_samples": 0,
            "write_in_flight": 0,
            "error": "",
            "source_sequence_available": False,
        }
    return {
        "format": "dex-native-v2",
        "format_version": 2,
        "episode_id": "synthetic-001",
        "state": "COMPLETED",
        "error": "",
        "quality_state": "good",
        "runtime": {"source_mode": "external"},
        "metadata": {"task_name": TASK["task_id"]},
        "selected_streams": list(streams),
        "skipped_streams": {},
        "streams": streams,
        "start_monotonic_ns": 10**9,
        "stop_monotonic_ns": 21 * 10**9,
    }


def synthetic_joint_message(stream):
    message = {
        "joint_names": stream["joint_names"],
        stream["position_field"]: [0.1] * len(stream["joint_names"]),
    }
    if stream["id"].startswith("wuji") and stream["role"] == "state":
        message.update(
            connected=True,
            enabled=True,
            position_valid=[True] * 20,
            online=[True] * 20,
            stale=[False] * 20,
            age_ms=[1.0] * 20,
            feedback_source="wuji_hand2",
        )
    if stream["id"] == "tianji_command":
        message.update(session_id="synthetic-session", valid_until={"sec": 101, "nanosec": 0})
    return message


def synthetic_snapshot():
    snapshot = {
        "collector": {"state": "IDLE", "collector_instance_id": "synthetic", "streams": []},
        "arrival_age_ms": {"collector": 0, "robot_state": 0},
        "publishers": {},
        "minimum_receive_hz": {},
        "baseline_source_gaps": {},
        "joint_messages": {},
        "post_warmup_status_count": 20,
        "ros_now_ns": 100 * 10**9,
        "robot_state": {
            "connected": True,
            "homed": True,
            "enabled": True,
            "feedback_valid": True,
            "command_allowed": True,
            "motion_allowed": True,
        },
    }
    for stream in TASK["streams"]:
        stream_id = stream["id"]
        snapshot["collector"]["streams"].append(
            {
                "stream_id": stream_id,
                "required": True,
                "enabled": True,
                "topic": stream["topic"],
                "message_type": stream["type"],
                "received": 100,
                "last_sample_age_ms": 1,
                "source_sequence_available": stream["id"] == "head_left_color",
                "source_sequence_gaps": 4,
            }
        )
        snapshot["publishers"][stream["topic"]] = [{"node_name": "synthetic"}]
        snapshot["minimum_receive_hz"][stream_id] = 30
        snapshot["baseline_source_gaps"][stream_id] = 4
        if stream["role"] != "image":
            snapshot["joint_messages"][stream_id] = synthetic_joint_message(stream)
            snapshot["arrival_age_ms"][stream_id] = 1
    return snapshot


class CollectionTests(unittest.TestCase):
    def test_contract_has_identical_state_action_order_and_both_hands(self):
        self.assertEqual(validate_task(TASK), [])
        task = copy.deepcopy(TASK)
        task["dataset"]["joint_names"].reverse()
        self.assertTrue(validate_task(task))

    def test_unmeasured_site_cannot_pass(self):
        errors = validate_site(TASK, load_document(TASK_DIR / "site.example.json"))
        self.assertGreater(len(errors), 10)
        self.assertEqual(validate_site(TASK, synthetic_site()), [])

    def test_site_rejects_boolean_domain_nonfinite_and_reversed_bounds(self):
        for field, value in (("ros_domain_id", True), ("ros_domain_id", 233)):
            site = synthetic_site()
            site[field] = value
            self.assertTrue(validate_site(TASK, site))
        site = synthetic_site()
        site["table"]["apple_x_bounds_m"] = [0.2, -0.2]
        site["prepare_pose_rad"]["left_arm"][0] = float("nan")
        self.assertGreaterEqual(len(validate_site(TASK, site)), 2)

    def test_profile_enforces_required_sources_and_preserves_qos_and_preview(self):
        profile = load_document(TASK_DIR / "collection.yaml")
        before = copy.deepcopy(profile)
        result = prepare_profile(TASK, profile, synthetic_site())
        required = {s["id"] for s in TASK["streams"]}
        self.assertEqual({key for key, value in result["streams"].items() if value["required"]}, required)
        self.assertTrue(all(result["streams"][key]["selected"] for key in required))
        self.assertFalse(result["streams"]["wrist_left_depth"]["selected"])
        self.assertFalse(result["streams"]["head_right_color"]["selected"])
        self.assertFalse(result["streams"]["pico_head_pose"]["selected"])
        self.assertEqual(profile, before)
        self.assertEqual(result["preview"], profile["preview"])
        for key, value in profile["streams"].items():
            self.assertEqual(result["streams"][key]["qos"], value["qos"])
        profile["streams"]["head_left_color"]["type"] = "sensor_msgs/msg/Image"
        with self.assertRaises(ValueError):
            prepare_profile(TASK, profile, synthetic_site())

    def test_shipped_profile_already_requires_all_training_streams(self):
        profile = load_document(TASK_DIR / "collection.yaml")
        required = {s["id"] for s in TASK["streams"]}
        self.assertEqual({key for key, value in profile["streams"].items() if value["required"]}, required)
        self.assertTrue(all(profile["streams"][key]["selected"] for key in required))
        for stream_id in ("head_right_color", "wrist_left_depth", "wrist_right_depth", "pico_head_pose"):
            self.assertFalse(profile["streams"][stream_id]["selected"])

    def test_teleop_overlay_enables_hands_without_changing_motion_limits(self):
        overlay = load_document(TASK_DIR / "teleop.yaml")
        self.assertEqual(
            overlay["whole_body_teleop"]["ros__parameters"], {"enable_hands": True, "head.enabled": False}
        )

    def test_storage_budget_includes_session_duration_margin_and_reserve(self):
        budget = storage_budget(TASK, synthetic_site(), 5)
        self.assertEqual(budget["raw_rgb_bytes_per_sec"], 176256000)
        self.assertGreater(budget["required_free_bytes"], budget["raw_rgb_bytes"])
        self.assertEqual(budget["raw_rgb_bytes"], 176256000 * 30 * 5)
        with self.assertRaises(ValueError):
            storage_budget(TASK, synthetic_site(), 0)

    def test_joint_mapping_uses_names_instead_of_input_order(self):
        stream = next(s for s in TASK["streams"] if s["id"] == "tianji_state_left")
        message = {"name": list(reversed(stream["joint_names"])), "position": list(range(7))}
        self.assertEqual(ordered_positions(message, stream), list(reversed(range(7))))
        message["name"][0] = "wrong_joint"
        with self.assertRaises(ValueError):
            ordered_positions(message, stream)

    def test_hand_retained_values_cannot_hide_invalid_stale_or_mock_feedback(self):
        stream = next(s for s in TASK["streams"] if s["id"] == "wuji_left_state")
        self.assertEqual(len(ordered_positions(synthetic_joint_message(stream), stream)), 20)
        mutations = [
            ("position_valid", [False] + [True] * 19),
            ("stale", [True] * 20),
            ("age_ms", [101.0] * 20),
            ("feedback_source", "dry_run"),
            ("connected", False),
            ("position", [float("inf")] * 20),
        ]
        for key, value in mutations:
            with self.subTest(key=key):
                message = synthetic_joint_message(stream)
                message[key] = value
                with self.assertRaises(ValueError):
                    ordered_positions(message, stream)

    def test_manifest_accepts_positive_control(self):
        self.assertTrue(audit_manifest(TASK, synthetic_manifest())["manifest_checks_passed"])

    def test_manifest_rejects_missing_stream_drops_mock_and_unfinished_writes(self):
        for mutation in ("missing", "drops", "pending", "source_gap", "mock", "wrong_task"):
            with self.subTest(mutation=mutation):
                manifest = synthetic_manifest()
                stream = manifest["streams"]["head_left_color"]
                if mutation == "missing":
                    manifest["selected_streams"].remove("head_left_color")
                elif mutation == "drops":
                    stream["dropped"] = 1
                elif mutation == "pending":
                    del stream["write_in_flight"]
                elif mutation == "source_gap":
                    stream.update(source_sequence_available=True, episode_source_sequence_gaps=1)
                elif mutation == "mock":
                    manifest["runtime"]["source_mode"] = "mock"
                else:
                    manifest["metadata"]["task_name"] = "another_task"
                self.assertFalse(audit_manifest(TASK, manifest)["manifest_checks_passed"])

    def test_optional_degradation_is_reported_without_claiming_sample_validity(self):
        manifest = synthetic_manifest()
        manifest["quality_state"] = "degraded"
        result = audit_manifest(TASK, manifest)
        self.assertTrue(result["manifest_checks_passed"])
        self.assertTrue(result["warnings"])
        self.assertIn("Manifest only", result["scope"])

    def test_annotation_requires_real_review_and_success_evidence(self):
        manifest = synthetic_manifest()
        annotation = make_annotation(TASK, manifest)
        self.assertTrue(validate_annotation(TASK, manifest, annotation))
        annotation.update(
            result="success",
            grasp_hand="left",
            apple_zone="left",
            bowl_zone="center",
            segment_start_offset_sec=1.0,
            segment_end_offset_sec=15.0,
            stable_in_bowl_sec=2.0,
            fully_released=True,
            human_intervention=False,
            reviewer="synthetic reviewer",
        )
        self.assertEqual(validate_annotation(TASK, manifest, annotation), [])
        annotation["segment_end_offset_sec"] = 21.0
        self.assertTrue(validate_annotation(TASK, manifest, annotation))
        annotation["segment_end_offset_sec"] = 15.0
        annotation["stable_in_bowl_sec"] = 1.0
        self.assertTrue(validate_annotation(TASK, manifest, annotation))
        manifest["episode_id"] = "different"
        self.assertTrue(validate_annotation(TASK, manifest, annotation))

    def test_split_is_reproducible_disjoint_and_covers_each_hand(self):
        records = [
            {
                "episode_id": f"{hand}-{i}",
                "grasp_hand": hand,
                "result": "success",
                "reviewer": "synthetic",
                "manifest_sha256": "0" * 64,
            }
            for hand in ("left", "right")
            for i in range(10)
        ]
        result = stratified_split(records, 1000)
        self.assertEqual(result, stratified_split(list(reversed(records)), 1000))
        self.assertEqual(len(result["validation_episode_ids"]), 2)
        self.assertTrue(any(i.startswith("left") for i in result["validation_episode_ids"]))
        self.assertTrue(any(i.startswith("right") for i in result["validation_episode_ids"]))
        self.assertFalse(set(result["train_episode_ids"]) & set(result["validation_episode_ids"]))
        with self.assertRaises(ValueError):
            stratified_split(records + [records[0]], 1000)
        with self.assertRaises(ValueError):
            stratified_split(records[:10], 1000)

    def test_preflight_connected_and_active_positive_controls(self):
        for active in (False, True):
            self.assertTrue(assess_snapshot(TASK, synthetic_snapshot(), active)["data_checks_passed"])

    def test_preflight_idle_commands_need_publishers_but_not_samples(self):
        snapshot = synthetic_snapshot()
        for source in snapshot["collector"]["streams"]:
            if source["stream_id"] in ("tianji_command", "wuji_left_command", "wuji_right_command"):
                source.update(received=0, last_sample_age_ms=-1)
                snapshot["joint_messages"].pop(source["stream_id"])
        self.assertTrue(assess_snapshot(TASK, snapshot, False)["data_checks_passed"])
        self.assertFalse(assess_snapshot(TASK, snapshot, True)["data_checks_passed"])

    def test_preflight_detects_wrong_profile_low_hz_staleness_duplicates_and_new_gaps(self):
        mutations = ["profile", "rate", "heartbeat", "publisher", "gap", "robot", "expiry"]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                snapshot = synthetic_snapshot()
                if mutation == "profile":
                    snapshot["collector"]["streams"][0]["required"] = False
                elif mutation == "rate":
                    snapshot["minimum_receive_hz"]["wrist_left_color"] = 15
                elif mutation == "heartbeat":
                    snapshot["arrival_age_ms"]["collector"] = 600
                elif mutation == "publisher":
                    snapshot["publishers"][TASK["streams"][0]["topic"]] *= 2
                elif mutation == "gap":
                    snapshot["collector"]["streams"][0]["source_sequence_gaps"] = 5
                elif mutation == "robot":
                    snapshot["robot_state"]["hardware_state_name"] = "HOLD"
                else:
                    snapshot["joint_messages"]["tianji_command"]["valid_until"]["sec"] = 99
                self.assertFalse(assess_snapshot(TASK, snapshot, True)["data_checks_passed"])

    def test_save_refuses_overwrite_and_session_cli_writes_frozen_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            site_path = root / "site.json"
            save_json(site_path, synthetic_site())
            with self.assertRaises(FileExistsError):
                save_json(site_path, {})
            output = root / "session"
            argv = [
                "collection_tools.py",
                "session",
                "--site",
                str(site_path),
                "--out",
                str(output),
                "--session-name",
                "synthetic-pilot",
                "--trials",
                "5",
            ]
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 0)
            session = json.loads((output / "session.json").read_text(encoding="utf-8"))
            self.assertEqual(session["platform_metadata_example"]["task_name"], TASK["task_id"])
            profile = load_document(output / "collection.generated.yaml")
            self.assertEqual(profile["collection"]["data_root"], "/synthetic/data")
            self.assertEqual(len((output / "trials.csv").read_text().splitlines()), 6)
            with patch("sys.argv", argv), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(), 1)


if __name__ == "__main__":
    unittest.main()
