import csv
import io
import zipfile
from typing import Dict, List

import numpy as np

from .data import Item


def clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def build_proxy_state(labels: Dict[str, int], text: str) -> Dict[str, float]:
    toxic = float(labels.get("toxic", 0))
    severe = float(labels.get("severe_toxic", 0))
    obscene = float(labels.get("obscene", 0))
    threat = float(labels.get("threat", 0))
    insult = float(labels.get("insult", 0))
    identity_hate = float(labels.get("identity_hate", 0))
    text_len = min(len(text), 500) / 500.0

    toxicity = clip01(0.75 * toxic + 0.25 * severe)
    harassment = clip01(0.8 * insult + 0.2 * toxic)
    hate = clip01(identity_hate)
    sexual = clip01(0.8 * obscene)
    self_harm = clip01(0.8 * threat)
    conflict_risk = clip01(max(insult, toxic, identity_hate))
    disagreement = clip01(0.4 * toxic + 0.4 * insult + 0.2 * text_len)
    escalation_level = clip01(0.45 * toxic + 0.35 * insult + 0.2 * threat)
    ambiguity = clip01(0.1 + 0.3 * (1.0 - severe) + 0.2 * text_len)
    uncertainty = clip01(0.15 + 0.25 * (1.0 - toxic) + 0.1 * text_len)

    return {
        "toxicity": toxicity,
        "harassment": harassment,
        "hate": hate,
        "sexual": sexual,
        "self_harm": self_harm,
        "conflict_risk": conflict_risk,
        "disagreement": disagreement,
        "escalation_level": escalation_level,
        "ambiguity": ambiguity,
        "uncertainty": uncertainty,
    }


def load_proxy_items_from_jigsaw_zip(zip_path: str, max_items: int) -> List[Item]:
    items: List[Item] = []
    with zipfile.ZipFile(zip_path, "r") as outer:
        if "train.csv" in outer.namelist():
            train_raw = outer.open("train.csv")
            train_stream = io.TextIOWrapper(train_raw, encoding="utf-8")
        else:
            if "train.csv.zip" not in outer.namelist():
                raise ValueError(f"Could not find train.csv or train.csv.zip inside {zip_path}")
            nested_bytes = outer.read("train.csv.zip")
            with zipfile.ZipFile(io.BytesIO(nested_bytes), "r") as inner:
                inner_csv_name = next((n for n in inner.namelist() if n.endswith(".csv")), None)
                if inner_csv_name is None:
                    raise ValueError("train.csv.zip does not contain a csv file")
                train_raw = inner.open(inner_csv_name)
                train_stream = io.TextIOWrapper(train_raw, encoding="utf-8")
                reader = csv.DictReader(train_stream)
                for i, row in enumerate(reader):
                    if i >= max_items:
                        break
                    text = row.get("comment_text", "")
                    labels = {
                        "toxic": int(row.get("toxic", 0)),
                        "severe_toxic": int(row.get("severe_toxic", 0)),
                        "obscene": int(row.get("obscene", 0)),
                        "threat": int(row.get("threat", 0)),
                        "insult": int(row.get("insult", 0)),
                        "identity_hate": int(row.get("identity_hate", 0)),
                    }
                    items.append(
                        Item(
                            id=str(row.get("id", i)),
                            text=text,
                            labels={"toxic": labels["toxic"]},
                            state=build_proxy_state(labels, text),
                        )
                    )
                if not items:
                    raise ValueError(f"No items built from zip file: {zip_path}")
                return items

        reader = csv.DictReader(train_stream)
        for i, row in enumerate(reader):
            if i >= max_items:
                break
            text = row.get("comment_text", "")
            labels = {
                "toxic": int(row.get("toxic", 0)),
                "severe_toxic": int(row.get("severe_toxic", 0)),
                "obscene": int(row.get("obscene", 0)),
                "threat": int(row.get("threat", 0)),
                "insult": int(row.get("insult", 0)),
                "identity_hate": int(row.get("identity_hate", 0)),
            }
            items.append(
                Item(
                    id=str(row.get("id", i)),
                    text=text,
                    labels={"toxic": labels["toxic"]},
                    state=build_proxy_state(labels, text),
                )
            )
    if not items:
        raise ValueError(f"No items built from zip file: {zip_path}")
    return items
