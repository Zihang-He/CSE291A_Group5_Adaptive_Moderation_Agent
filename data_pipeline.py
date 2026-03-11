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
import io
from dataclasses import asdict
from pathlib import Path
from typing import AsyncIterator, Dict, Iterable, Optional
import zipfile

from perception import (
    ContentContext,
    EngagementSignals,
    MediaFeatures,
    PerceptionAgent,
    PerceptionInput,
    PerceptionState,
)


DATA_DIR = Path(__file__).parent
KAGGLE_BUNDLE = "jigsaw-toxic-comment-classification-challenge.zip"


def _iter_csv_rows_from_text_io(text_io: io.TextIOBase) -> Iterable[Dict[str, str]]:
    reader = csv.DictReader(text_io)
    for row in reader:
        yield row


def _iter_rows_from_csv_file(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        yield from _iter_csv_rows_from_text_io(f)


def _iter_rows_from_csv_zip(zip_path: Path, csv_name: str) -> Iterable[Dict[str, str]]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        if csv_name not in names:
            raise FileNotFoundError(f"{csv_name} not found in {zip_path}")
        with zf.open(csv_name, "r") as raw_f:
            with io.TextIOWrapper(raw_f, encoding="utf-8") as text_f:
                yield from _iter_csv_rows_from_text_io(text_f)


def _iter_rows_from_kaggle_bundle(bundle_path: Path, csv_name: str) -> Iterable[Dict[str, str]]:
    nested_zip_name = f"{csv_name}.zip"
    with zipfile.ZipFile(bundle_path, "r") as outer_zf:
        names = set(outer_zf.namelist())
        if nested_zip_name not in names:
            raise FileNotFoundError(f"{nested_zip_name} not found in {bundle_path}")
        nested_bytes = outer_zf.read(nested_zip_name)

    with zipfile.ZipFile(io.BytesIO(nested_bytes), "r") as nested_zf:
        nested_names = set(nested_zf.namelist())
        if csv_name not in nested_names:
            raise FileNotFoundError(f"{csv_name} not found in nested {nested_zip_name}")
        with nested_zf.open(csv_name, "r") as raw_f:
            with io.TextIOWrapper(raw_f, encoding="utf-8") as text_f:
                yield from _iter_csv_rows_from_text_io(text_f)


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

    csv_path = DATA_DIR / fname
    if csv_path.exists():
        yield from _iter_rows_from_csv_file(csv_path)
        return

    csv_zip_path = DATA_DIR / f"{fname}.zip"
    if csv_zip_path.exists():
        yield from _iter_rows_from_csv_zip(csv_zip_path, fname)
        return

    bundle_path = DATA_DIR / KAGGLE_BUNDLE
    if bundle_path.exists():
        yield from _iter_rows_from_kaggle_bundle(bundle_path, fname)
        return

    # As a final fallback, search recursively in the project root.
    matches = list(DATA_DIR.rglob(fname))
    if matches:
        yield from _iter_rows_from_csv_file(matches[0])
        return

    raise FileNotFoundError(
        f"CSV file not found: {csv_path}. "
        f"Tried plain CSV, {fname}.zip, and {KAGGLE_BUNDLE}."
    )


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


