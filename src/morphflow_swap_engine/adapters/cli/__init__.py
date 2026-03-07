from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="morphflow-swap-engine")
    parser.add_argument("--source-face", required=True, help="Path to the reference face image.")
    parser.add_argument("--target", required=True, help="Path to the target video.")
    parser.add_argument("--output", help="Path for the processed output video.")
    parser.add_argument(
        "--profile",
        default="balanced",
        choices=("balanced", "high_quality", "throughput_max"),
        help="Runtime profile.",
    )
    parser.add_argument("--label", default="", help="Optional label for the reference face.")
    parser.add_argument("--config", help="Optional path to an engine INI file.")
    parser.add_argument("--json", action="store_true", help="Print the response payload as JSON.")
    return parser


def _build_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "source_face_path": str(Path(args.source_face)),
        "target_path": str(Path(args.target)),
        "profile": args.profile,
        "label": args.label,
    }
    if args.output:
        payload["output_path"] = str(Path(args.output))
    if args.config:
        payload["config_path"] = str(Path(args.config))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    from morphflow_swap_engine.adapters.morphflow import MorphFlowAdapter

    response = MorphFlowAdapter().handle(_build_payload(args))
    if args.json:
        print(json.dumps(response, indent=2))
    else:
        status = "success" if response["success"] else "failure"
        print(f"status: {status}")
        print(f"output: {response['output_path']}")
        print(f"frames: {response['frames_processed']}")
        print(f"duration_seconds: {response['duration_seconds']:.3f}")
        if response["error_message"]:
            print(f"error: {response['error_message']}")
    return 0 if response["success"] else 1


__all__ = ["build_parser", "main"]
