#!/usr/bin/env python
"""Convenience test runner for the robot project.

Usage:
  python scripts/run_tests.py              Run all fast tests
  python scripts/run_tests.py --all        Run all tests (including slow/e2e)
  python scripts/run_tests.py --unit       Run unit tests only
  python scripts/run_tests.py --e2e        Run E2E tests only
  python scripts/run_tests.py --sim        Run simulation tests only
  python scripts/run_tests.py --cov        Run with coverage report
"""

import sys
import os
import subprocess
import argparse

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")


def run(args: list[str]) -> int:
    """Run pytest with given arguments."""
    cmd = [sys.executable, "-m", "pytest"] + args
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=PROJECT_ROOT).returncode


def main():
    parser = argparse.ArgumentParser(description="Robot project test runner")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Run all tests including slow/e2e")
    group.add_argument("--unit", action="store_true", help="Unit tests only")
    group.add_argument("--e2e", action="store_true", help="E2E tests only")
    group.add_argument("--sim", action="store_true", help="Simulation tests only")
    parser.add_argument("--cov", action="store_true", help="Generate coverage report")
    parser.add_argument("-k", "--keyword", type=str, help="Filter tests by keyword expression")
    args = parser.parse_args()

    pytest_args = []

    if args.unit:
        pytest_args.append("tests/unit/")
    elif args.e2e:
        pytest_args.append("tests/e2e/")
    elif args.sim:
        pytest_args.append("tests/simulation/")
    else:
        pytest_args.append("tests/")

    if not args.all:
        pytest_args.extend(["-m", "not slow and not gpu and not e2e"])

    if args.cov:
        pytest_args.extend([
            "--cov=control", "--cov=perception", "--cov=simulation",
            "--cov-report=term-missing",
        ])

    if args.keyword:
        pytest_args.extend(["-k", args.keyword])

    sys.exit(run(pytest_args))


if __name__ == "__main__":
    main()
