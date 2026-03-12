"""
Run the ReAct action chooser over saved perception outputs.

Example:
    python run_action_chooser_on_output.py
"""
from __future__ import annotations

import json
from pathlib import Path

from action_chooser import build_react_action_chooser_from_env


def main(
    input_path: str = "jigsaw_perception_output.jsonl",
    output_path: str = "jigsaw_action_output.jsonl",
) -> None:
    chooser = build_react_action_chooser_from_env()
    in_path = Path(input_path)
    out_path = Path(output_path)

    count = 0
    with in_path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            state = row.get("state") or {}
            decision = chooser.choose_action(state)

            row["action_decision"] = {
                "action_id": decision.action_id,
                "action_name": decision.action_name,
                "reasoning": decision.reasoning,
                "used_fallback": decision.used_fallback,
                "trace": decision.trace,
            }
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")

            count += 1
            if count % 10 == 0:
                print(f"Processed {count} rows...")

    print(f"Done. Wrote {count} rows to {out_path}")


if __name__ == "__main__":
    main()
