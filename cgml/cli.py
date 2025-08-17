#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from typing import Optional, List

from .loader import load_cgml_file
from .simulator import GameSimulator


def simulate(argv: Optional[List[str]] = None) -> None:
    """CLI entrypoint: simulate a CGML YAML file.

    Usage: simulate path/to/game.yml [-p PLAYERS]
    """
    parser = argparse.ArgumentParser(prog="simulate", description="Simulate a CGML YAML game file")
    parser.add_argument("cgml_file", help="Path to the CGML YAML file to load")
    parser.add_argument(
        "-p",
        "--players",
        type=int,
        default=None,
        help="Number of players (defaults to value from the CGML file)",
    )
    args = parser.parse_args(argv)

    cgml = load_cgml_file(args.cgml_file)
    if cgml is None:
        print("Failed to load CGML file.")
        sys.exit(1)

    player_count = args.players if args.players is not None else cgml.meta.players.max
    simulator = GameSimulator(cgml, player_count=player_count)
    simulator.run()


if __name__ == "__main__":
    simulate()
