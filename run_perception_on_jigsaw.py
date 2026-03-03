"""
CLI script to run the perception module over the Jigsaw toxic comment dataset.

This is useful for:
- producing a dataset of PerceptionState features paired with Jigsaw labels
- sanity-checking how the perception agent behaves at scale
"""

import asyncio
import json
import os
from typing import Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI

from data_pipeline import perceive_jigsaw_stream
from perception import PerceptionAgent, SYSTEM_PROMPT


async def main(
    split: str = "train",
    limit: Optional[int] = 100,
    output_path: str = "jigsaw_perception_output.jsonl",
) -> None:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("MOD_AGENT_OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL", "api-gpt-oss-120b")
    if not api_key:
        raise RuntimeError(
            "No OpenAI API key configured. Set OPENAI_API_KEY or MOD_AGENT_OPENAI_API_KEY."
        )

    base_url = os.getenv("OPENAI_BASE_URL", "https://tritonai-api.ucsd.edu")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    agent = PerceptionAgent(client, model_name)

    print(f"Running perception on Jigsaw split='{split}' limit={limit} model={model_name}")
    out_f = open(output_path, "w", encoding="utf-8")
    n = 0
    async for item in perceive_jigsaw_stream(agent, split=split, limit=limit):
        out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
        n += 1
        if n % 10 == 0:
            print(f"Processed {n} rows...")
    out_f.close()
    print(f"Done. Wrote {n} rows to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())

