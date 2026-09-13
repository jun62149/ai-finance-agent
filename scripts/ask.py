"""Ask the Agent a question from the command line (Phase 7).

Usage:
    python scripts/ask.py "分析一下京东方A"
    python scripts/ask.py "现在评分最高的10只股票" --json
    python scripts/ask.py "贵州茅台最近有什么公告" --show-trace

Requires LLM_API_KEY in .env. Tool errors and missing data are reported as-is —
the Agent never fills gaps with invented content.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.llm import LLMClient
from app.agents.graph import run_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the investment-research Agent")
    parser.add_argument("query")
    parser.add_argument("--json", action="store_true", help="print the raw state as JSON")
    parser.add_argument("--show-trace", action="store_true", help="print intent/plan/tools")
    args = parser.parse_args()

    llm = LLMClient()
    print(f"model={llm.model} base_url={llm.base_url} key_set={llm.available}")
    started = time.time()
    state = run_agent(args.query, llm=llm)
    elapsed = time.time() - started

    if args.json:
        print(json.dumps(state, ensure_ascii=False, indent=2, default=str))
        return

    if args.show_trace or True:
        print("")
        print(f"intent={state.get('intent')} (source={state.get('intent_source')}) "
              f"stocks={state.get('stock_codes')}")
        print(f"plan={state.get('plan')}")
        for call in state.get("tool_calls") or []:
            print(f"  tool {call['name']} ok={call['ok']} args={call['arguments']}")
        for error in state.get("errors") or []:
            print(f"  [!] {error.get('code')}: {error.get('message')}")
        for note in state.get("evidence_notes") or []:
            print(f"  [note] {note}")
        print(f"evidence={len(state.get('evidence') or [])} "
              f"evidence_ok={state.get('evidence_ok')} elapsed={elapsed:.1f}s")

    print("")
    print(state.get("final_answer") or "(no answer)")


if __name__ == "__main__":
    main()
