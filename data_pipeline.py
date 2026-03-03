"""
Dataset utilities for integrating the Jigsaw toxic comment dataset with the
perception pipeline.

This module provides:
- lightweight CSV loading for the Jigsaw train/test splits
- a generator that feeds rows into the PerceptionAgent and yields PerceptionState

The goal is to make it easy to:
- run the perception module over a large corpus
- log the resulting structured states for downstream training / evaluation
"""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import AsyncIterator, Dict, Iterable, Optional

from perception import (
    ContentContext,
    EngagementSignals,
    MediaFeatures,
    PerceptionAgent,
    PerceptionInput,
    PerceptionState,
)


DATA_DIR = Path(__file__).parent


def iter_jigsaw_rows(split: str = "train") -> Iterable[Dict[str, str]]:
    """
    Stream rows from the Jigsaw toxic comment CSV.

    Args:
        split: "train" or "test".
    """
    fname = {
        "train": "train.csv",
        "test": "test.csv",
    }.get(split)
    if fname is None:
        raise ValueError(f"Unknown split '{split}', expected 'train' or 'test'.")

    path = DATA_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


async def perceive_jigsaw_stream(
    agent: PerceptionAgent,
    split: str = "train",
    limit: Optional[int] = None,
) -> AsyncIterator[Dict[str, object]]:
    """
    Run the perception agent over a stream of Jigsaw rows.

    Yields dicts that contain:
        {
          "id": ...,
          "comment_text": ...,
          "labels": {...},         # present only for train split
          "state": {...},          # PerceptionState as plain dict
        }
    """
    count = 0
    for row in iter_jigsaw_rows(split=split):
        if limit is not None and count >= limit:
            break
        count += 1

        comment_text = row.get("comment_text", "")

        # Map Jigsaw labels (0/1) into a label dict, only for train.
        labels: Dict[str, int] = {}
        if split == "train":
            for key in [
                "toxic",
                "severe_toxic",
                "obscene",
                "threat",
                "insult",
                "identity_hate",
            ]:
                v = row.get(key)
                try:
                    labels[key] = int(v) if v is not None and v != "" else 0
                except ValueError:
                    labels[key] = 0

        inp = PerceptionInput(
            post_text=comment_text,
            media=MediaFeatures(),  # no images in Jigsaw; keep empty
            context=ContentContext(),  # no thread context in Jigsaw; keep empty
            engagement=EngagementSignals(),  # can be filled with synthetic signals later
        )

        state: PerceptionState = await agent.perceive(inp)

        yield {
            "id": row.get("id"),
            "comment_text": comment_text,
            "labels": labels if labels else None,
            "state": asdict(state),
        }


