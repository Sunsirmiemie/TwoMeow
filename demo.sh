#!/usr/bin/env bash
set -euo pipefail

PYTHON=".venv/bin/python"

echo "════════════════════════════════════════════════════════════════════════"
echo "  TwoMeow — TechJam 2026 Conversational Search Agent"
echo "  Public set: 200 sessions  |  No LLM  |  Fully offline"
echo "════════════════════════════════════════════════════════════════════════"

echo ""
echo "[ Overall evaluation — 200 sessions ]"
echo ""
PYTHONHASHSEED=0 $PYTHON scripts/run_public_eval.py --no-dense --output results.json

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "  Scenario traces"
echo "════════════════════════════════════════════════════════════════════════"

echo ""
echo "[ 1/4  BUYING — hard constraint from turn 1 ]"
PYTHONHASHSEED=0 $PYTHON scripts/trace_session.py --id public_0010

echo ""
echo "[ 2/4  BROWSING — vague intent, guided to target ]"
PYTHONHASHSEED=0 $PYTHON scripts/trace_session.py --id public_0019

echo ""
echo "[ 3/4  INTENT OVERRIDE — agent resets on preference change ]"
PYTHONHASHSEED=0 $PYTHON scripts/trace_session.py --id public_0072

echo ""
echo "[ 4/4  BOUNDARY — no user preference, agent still finds target ]"
PYTHONHASHSEED=0 $PYTHON scripts/trace_session.py --id public_0104
