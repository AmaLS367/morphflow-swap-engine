from __future__ import annotations

from morphflow_swap_engine.adapters.cli import build_parser


def test_cli_parser_maps_arguments() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--source-face",
            "source.jpg",
            "--target",
            "target.mp4",
            "--profile",
            "high_quality",
            "--config",
            "engine.ini",
            "--json",
        ]
    )

    assert args.source_face == "source.jpg"
    assert args.target == "target.mp4"
    assert args.profile == "high_quality"
    assert args.config == "engine.ini"
    assert args.json is True
