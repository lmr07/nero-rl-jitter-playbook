#!/usr/bin/env python3
"""Evaluate the quantitative release gates for a model39000 continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "training_profiles/model39000_woimu_training_identification.json"
DIRECTIONS = ("W", "S", "Z", "C", "A", "D")


def _load_directory(path: Path) -> dict[str, dict]:
    rows = {}
    for direction in DIRECTIONS:
        source = path / f"{direction}.json"
        if not source.is_file():
            raise FileNotFoundError(f"missing audit result: {source}")
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("schema") != "model39000_woimu.isaacgym_observation_action_audit.v1":
            raise ValueError(f"unexpected audit schema in {source}")
        rows[direction] = payload
    return rows


def _weighted_mean(rows: dict[str, dict], key: str) -> float:
    counts = np.asarray([float(rows[d]["num_envs"]) for d in DIRECTIONS])
    values = np.asarray([float(rows[d][key]) for d in DIRECTIONS])
    return float(np.average(values, weights=counts))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identified-dir", type=Path, required=True)
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    parser.add_argument("--candidate-nominal-dir", type=Path)
    parser.add_argument("--source-nominal-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (args.candidate_nominal_dir is None) != (args.source_nominal_dir is None):
        parser.error(
            "--candidate-nominal-dir and --source-nominal-dir must be supplied together"
        )

    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    acceptance = profile[f"stage{args.stage}"]["acceptance"]
    identified = _load_directory(args.identified_dir)

    samples_per_direction = {
        d: int(identified[d]["num_envs"]) for d in DIRECTIONS
    }
    sample_floor = int(
        acceptance.get("seeds_per_direction_min", 20 if args.stage == 2 else 20)
    )
    sample_ok = all(value >= sample_floor for value in samples_per_direction.values())

    all_ee_p2p = np.concatenate(
        [
            np.asarray(
                identified[d]["walking_ee_position_p2p_mm_by_env"],
                dtype=np.float64,
            )
            for d in DIRECTIONS
        ]
    )
    ee_p95 = float(np.percentile(all_ee_p2p, 95.0))
    ee_limit = float(acceptance["walking_ee_position_p2p_p95_mm_max"])
    ee_ok = ee_p95 < ee_limit

    direction_step_p99 = {
        d: float(identified[d]["nero_target_group_p99_abs_step_rad"])
        for d in DIRECTIONS
    }
    # The maximum of the six per-direction p99 values is conservative versus
    # pooling all samples and prevents one command direction being hidden by
    # five easier ones.
    target_step_p99 = max(direction_step_p99.values())
    target_step_limit = float(acceptance["nero_actor_target_step_p99_rad_max"])
    target_step_ok = target_step_p99 < target_step_limit

    identified_success = {
        d: float(identified[d]["gait_success_rate"]) for d in DIRECTIONS
    }
    relative_checks = None
    relative_ok = False
    if args.candidate_nominal_dir is not None:
        candidate = _load_directory(args.candidate_nominal_dir)
        source = _load_directory(args.source_nominal_dir)
        success_floor = float(
            acceptance.get(
                "nominal_success_rate_relative_to_source_min",
                acceptance.get("nominal_success_rate_relative_to_stage1_min", 0.95),
            )
        )
        reward_floor = float(
            acceptance.get(
                "command_tracking_reward_relative_to_source_min",
                acceptance.get("command_tracking_reward_relative_to_stage1_min", 0.95),
            )
        )
        by_direction = {}
        for direction in DIRECTIONS:
            candidate_success = float(candidate[direction]["gait_success_rate"])
            source_success = float(source[direction]["gait_success_rate"])
            candidate_reward = float(
                candidate[direction]["command_tracking_reward_mean"]
            )
            source_reward = float(source[direction]["command_tracking_reward_mean"])
            success_ratio = (
                1.0 if source_success <= 1.0e-12
                else candidate_success / source_success
            )
            reward_ratio = (
                1.0 if source_reward <= 1.0e-12
                else candidate_reward / source_reward
            )
            by_direction[direction] = {
                "success_ratio": success_ratio,
                "command_tracking_reward_ratio": reward_ratio,
                "passed": success_ratio >= success_floor and reward_ratio >= reward_floor,
            }
        relative_ok = all(row["passed"] for row in by_direction.values())
        relative_checks = {
            "success_ratio_floor": success_floor,
            "command_tracking_reward_ratio_floor": reward_floor,
            "candidate_success_mean": _weighted_mean(candidate, "gait_success_rate"),
            "source_success_mean": _weighted_mean(source, "gait_success_rate"),
            "candidate_command_tracking_reward_mean": _weighted_mean(
                candidate, "command_tracking_reward_mean"
            ),
            "source_command_tracking_reward_mean": _weighted_mean(
                source, "command_tracking_reward_mean"
            ),
            "by_direction": by_direction,
            "passed": relative_ok,
        }

    release_eligible = sample_ok and ee_ok and target_step_ok and relative_ok
    result = {
        "schema": "model39000_woimu.zeroshot_release_gate.v1",
        "stage": args.stage,
        "identified_dir": str(args.identified_dir.resolve()),
        "samples_per_direction": samples_per_direction,
        "sample_floor": sample_floor,
        "sample_gate_passed": sample_ok,
        "identified_success_rate_by_direction": identified_success,
        "walking_ee_base_relative_p2p_p95_mm": ee_p95,
        "walking_ee_base_relative_p2p_p95_limit_mm": ee_limit,
        "walking_ee_gate_passed": ee_ok,
        "nero_target_step_p99_rad_by_direction": direction_step_p99,
        "nero_target_step_p99_rad": target_step_p99,
        "nero_target_step_p99_limit_rad": target_step_limit,
        "nero_target_step_gate_passed": target_step_ok,
        "relative_nominal_checks": relative_checks,
        "release_eligible": release_eligible,
    }
    output = args.output or args.identified_dir / "release_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if relative_checks is None:
        print(
            "INCOMPLETE: nominal candidate/source directories are required for release.",
            file=sys.stderr,
        )
    return 0 if release_eligible else 2


if __name__ == "__main__":
    raise SystemExit(main())
