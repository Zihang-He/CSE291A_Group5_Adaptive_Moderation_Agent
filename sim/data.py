# sim/data.py
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Item:
    id: str
    text: str
    labels: Dict[str, int]
    state: Dict[str, Any]


def load_items(jsonl_path: str) -> List[Item]:
    items: List[Item] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            if obj.get("state") is None:
                continue

            items.append(
                Item(
                    id=str(obj.get("id", "")),
                    text=str(obj.get("comment_text", "")),
                    labels=obj.get("labels") or {},
                    state=obj.get("state") or {},
                )
            )
    if not items:
        raise ValueError(f"No valid items with state found in {jsonl_path}")
    return items