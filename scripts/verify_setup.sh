#!/usr/bin/env bash
# Milestone 1 check: tests + offline dry run on real Banking77 (mock teacher, tf-idf student).
set -euo pipefail
pytest -q
homeroom run --config configs/pilot_banking77.yaml --dry-run --seeds 0
homeroom check-gpu || true
