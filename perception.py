"""
Perception module for the Adaptive Moderation Agent.

Stage 1 (Perception) takes:
- content (text + optional image/video placeholder features)
- platform / engagement signals
- optional conversation context

and uses a multimodal‑aware LLM to produce a structured state representation
that will later feed into the decision & learning (control) stage.
"""
import asyncio
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI


SYSTEM_PROMPT = """You are a content moderation perception module for a social media platform.

You are the first stage of a two-stage moderation agent (Perception -> Control).
Your job is to READ the full input (content + context + engagement signals)
and output a STRICT JSON object that summarizes the state of this item
for downstream decision-making.

=== INPUT FIELDS ===
- post.text: the main text of the post
- post.media_summary: short description of images / video if available (can be empty)
- context.thread_summary: short natural language summary of prior thread or conversation
- context.parent_text: direct parent comment/post if this is a reply
- engagement: platform and user behavior signals, including:
    - likes, shares, comments, reports, reply_depth, engagement_growth_rate
    - avg_comment_sentiment (from -1..1), report_rate, etc.

You NEVER see raw user identifiers, only anonymized behavior summaries.

=== OUTPUT SCHEMA (STRICT JSON) ===
{
  "toxicity": number,              // 0..1
  "harassment": number,            // 0..1
  "hate": number,                  // 0..1
  "self_harm": number,             // 0..1
  "sexual": number,                // 0..1

  "conflict_risk": number,         // 0..1  (likelihood the thread escalates)
  "escalation_level": number,      // 0..1  (how escalated the thread already is)
  "ambiguity": number,             // 0..1  (how unclear / needs more context)
  "uncertainty": number,           // 0..1  (your confidence inverse, higher = less sure)

  "disagreement": number,          // 0..1  (how much ideological / factual disagreement)

  "engagement_risk": number,       // 0..1  (how much current engagement profile suggests trouble
                                      // e.g., high reports, high growth with toxic comments)

  "top_reasons": [string, string, string],  // up to 3 short phrases

  "suggested_action": string,      // one of: "do_nothing" | "downrank" | "add_friction" | "throttle"

  "notes": string                  // optional short natural language rationale (<= 3 sentences)
}

=== RULES ===
- Output valid JSON only. No markdown. No comments.
- All numbers must be decimals between 0 and 1.
- If you are unsure, increase "uncertainty" and lower other scores.
- Keep "top_reasons" short phrases.
- Always include all fields in the schema.
"""


# -----------------------------
# Data structures
# -----------------------------


@dataclass
class EngagementSignals:
    """Aggregated platform signals for a single content item."""

    likes: int = 0
    shares: int = 0
    comments: int = 0
    reports: int = 0
    reply_depth: int = 0

    # Derived / normalized signals (0..1 are convenient for the LLM)
    engagement_growth_rate: float = 0.0  # 0..1, how fast engagement is increasing
    report_rate: float = 0.0  # 0..1, fraction of interactions that are reports
    avg_comment_sentiment: float = 0.0  # -1..1, where -1 is very negative, +1 is very positive


@dataclass
class ContentContext:
    """Conversation context available to perception."""

    thread_summary: str = ""
    parent_text: str = ""


@dataclass
class MediaFeatures:
    """
    Lightweight representation of visual/audio modality.

    For now we assume another component (e.g., CLIP/vision encoder or
    an LLM with vision) produces a short natural language summary
    and optional safety tags. This keeps the perception agent
    architecture-friendly without forcing a specific vision backend.
    """

    summary: str = ""  # a short textual description of the media
    safety_tags: List[str] = field(default_factory=list)  # e.g., ["weapon", "blood"]


@dataclass
class PerceptionInput:
    """Full input to the perception module."""

    post_text: str
    media: MediaFeatures = field(default_factory=MediaFeatures)
    context: ContentContext = field(default_factory=ContentContext)
    engagement: EngagementSignals = field(default_factory=EngagementSignals)


@dataclass
class PerceptionState:
    """Structured state representation for downstream control/learning."""

    toxicity: float
    harassment: float
    hate: float
    self_harm: float
    sexual: float

    conflict_risk: float
    escalation_level: float
    ambiguity: float
    uncertainty: float
    disagreement: float
    engagement_risk: float

    top_reasons: List[str]
    suggested_action: str
    notes: str

    @staticmethod
    def from_raw_json(data: Dict[str, Any]) -> "PerceptionState":
        """
        Validate and coerce a raw JSON dict into a PerceptionState.
        This is a light check so that downstream code can rely
        on the presence of all fields.
        """

        def as_float(key: str, default: float = 0.0) -> float:
            try:
                v = float(data.get(key, default))
            except (TypeError, ValueError):
                v = default
            # Clamp to [0, 1] where appropriate
            if key in {
                "toxicity",
                "harassment",
                "hate",
                "self_harm",
                "sexual",
                "conflict_risk",
                "escalation_level",
                "ambiguity",
                "uncertainty",
                "disagreement",
                "engagement_risk",
            }:
                v = max(0.0, min(1.0, v))
            return v

        top_reasons = data.get("top_reasons") or []
        if not isinstance(top_reasons, list):
            top_reasons = [str(top_reasons)]
        top_reasons = [str(r) for r in top_reasons][:3]

        return PerceptionState(
            toxicity=as_float("toxicity"),
            harassment=as_float("harassment"),
            hate=as_float("hate"),
            self_harm=as_float("self_harm"),
            sexual=as_float("sexual"),
            conflict_risk=as_float("conflict_risk"),
            escalation_level=as_float("escalation_level"),
            ambiguity=as_float("ambiguity"),
            uncertainty=as_float("uncertainty"),
            disagreement=as_float("disagreement"),
            engagement_risk=as_float("engagement_risk"),
            top_reasons=top_reasons,
            suggested_action=str(data.get("suggested_action", "do_nothing")),
            notes=str(data.get("notes", "")),
        )


# -----------------------------
# Perception agent (OpenAI/AsyncOpenAI client wrapper)
# -----------------------------


class PerceptionAgent:
    def __init__(self, client: AsyncOpenAI, model_name: str) -> None:
        self._client = client
        self._model_name = model_name

    async def perceive(self, inp: PerceptionInput) -> PerceptionState:
        """
        High-level API: given a PerceptionInput, return a PerceptionState.

        This is what the control / RL agent will consume.
        """
        user_payload: Dict[str, Any] = {
            "post": {
                "text": inp.post_text,
                "media_summary": inp.media.summary,
                "media_safety_tags": inp.media.safety_tags,
            },
            "context": {
                "thread_summary": inp.context.thread_summary,
                "parent_text": inp.context.parent_text,
            },
            "engagement": asdict(inp.engagement),
        }

        raw = await self._chat_json(user_payload)
        return PerceptionState.from_raw_json(raw)

    async def _chat_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Low-level helper: prompt the model and parse the strict JSON response.
        Uses the official OpenAI client under the hood.
        """
        prompt = (
            "You will receive a JSON object describing a single content item "
            "on a social media platform.\n"
            "Carefully read all fields and then respond with a STRICT JSON object "
            "matching the specified output schema.\n\n"
            f"INPUT:\n{json.dumps(payload, ensure_ascii=False)}\n\n"
            "Return ONLY the JSON object. No extra text."
        )

        # Call the async OpenAI client directly (works with Triton proxy too).
        resp = await self._client.chat.completions.create(
            model=self._model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        text = resp.choices[0].message.content

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            # In a research prototype it's often better to fail fast;
            # later you can add a repair loop if needed.
            raise ValueError(f"Model did not return valid JSON:\n{text}") from e

        return data


# -----------------------------
# Simple CLI for debugging
# -----------------------------


async def main() -> None:
    """
    Minimal CLI to poke the perception agent.

    - Type a post.
    - Optionally set context and engagement fields via simple prompts.
    - See the structured state JSON printed to stdout.
    """

    # Load .env if present so you can keep keys out of code.
    load_dotenv()

    api_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("MOD_AGENT_OPENAI_API_KEY")
    )
    model_name = os.getenv("OPENAI_MODEL", "api-gpt-oss-120b")

    if not api_key:
        raise RuntimeError(
            "No OpenAI API key configured.\n"
            "Set one of: OPENAI_API_KEY or MOD_AGENT_OPENAI_API_KEY.\n"
            "You can put it in a .env file at the project root."
        )

    # Allow overriding base URL so you can use the Triton proxy, etc.
    base_url = os.getenv("OPENAI_BASE_URL", "https://tritonai-api.ucsd.edu")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    ai = PerceptionAgent(client, model_name)

    print(f"Perception agent ready. Model={model_name}")
    print(
        "Type a post. Prefix with 'CTX:' to set thread summary, "
        "'PARENT:' for parent text. Ctrl+C to quit.\n"
    )

    ctx = ContentContext()
    engagement = EngagementSignals()

    while True:
        post = input("POST> ").strip()
        if not post:
            continue

        # Quick-and-dirty context setter
        if post.startswith("CTX:"):
            ctx.thread_summary = post[len("CTX:") :].strip()
            print("Thread summary set.\n")
            continue
        if post.startswith("PARENT:"):
            ctx.parent_text = post[len("PARENT:") :].strip()
            print("Parent text set.\n")
            continue

        # For now we keep engagement simple; you can wire this up to logs later.
        try:
            raw_likes = input(f"likes (current {engagement.likes})> ").strip()
            if raw_likes:
                engagement.likes = int(raw_likes)
            raw_comments = input(f"comments (current {engagement.comments})> ").strip()
            if raw_comments:
                engagement.comments = int(raw_comments)
            raw_reports = input(f"reports (current {engagement.reports})> ").strip()
            if raw_reports:
                engagement.reports = int(raw_reports)
            raw_sent = input(
                f"avg_comment_sentiment -1..1 (current {engagement.avg_comment_sentiment})> "
            ).strip()
            if raw_sent:
                engagement.avg_comment_sentiment = float(raw_sent)
        except ValueError:
            print("Invalid numeric input; keeping previous engagement values.\n")

        inp = PerceptionInput(
            post_text=post,
            media=MediaFeatures(),  # fill this if you have a media encoder
            context=ctx,
            engagement=engagement,
        )

        state = await ai.perceive(inp)
        # Print raw dict so you can easily pipe/parse downstream.
        print(json.dumps(asdict(state), ensure_ascii=False, indent=2))
        print()


if __name__ == "__main__":
    asyncio.run(main())
