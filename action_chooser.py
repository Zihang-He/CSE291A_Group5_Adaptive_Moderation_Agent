"""
ReAct-style action chooser for Stage 2 moderation control.

This module takes a structured PerceptionState (or plain dict with the same
fields) and chooses one of the supported moderation actions:
    0 -> do_nothing
    1 -> downrank
    2 -> add_friction
    3 -> throttle

The chooser runs a small ReAct loop:
1. The model inspects the structured state and current platform signals.
2. It may call local tools to derive a risk summary or inspect action tradeoffs.
3. It finishes with a single moderation action plus rationale.

If the model call fails or produces invalid output, the chooser falls back to a
deterministic rule-based policy so downstream code keeps working.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency in this repo
    def load_dotenv() -> bool:
        return False

ACTION_ID_TO_NAME = {
    0: "do_nothing",
    1: "downrank",
    2: "add_friction",
    3: "throttle",
}

ACTION_NAME_TO_ID = {name: idx for idx, name in ACTION_ID_TO_NAME.items()}

ACTION_DESCRIPTIONS = {
    "do_nothing": "Leave content untouched. Lowest user cost, highest spread risk.",
    "downrank": "Reduce distribution mildly. Good for moderate risk or early warning signs.",
    "add_friction": "Insert warning/friction before interaction. Good for high conflict or meaningful harm risk.",
    "throttle": "Strongly limit visibility/spread. Use for severe or rapidly escalating risk.",
}


SYSTEM_PROMPT = """You are the Stage 2 moderation controller for a social media platform.

You receive a structured safety evaluation from an upstream perception agent.
Your job is NOT to rescore the content from scratch. Your job is to choose the
best moderation action from the existing action space by using ReAct:

1. Think about what information is still needed.
2. Choose one local tool at a time.
3. Read the tool observation.
4. Finish with exactly one action.

Available actions:
- do_nothing
- downrank
- add_friction
- throttle

Available tools:
- summarize_risk: derive compact harm/conflict/escalation summary from the state
- inspect_action_space: inspect action tradeoffs and default decision bands

Decision principles:
- Prefer stronger intervention when harm is severe.
- Prefer friction/downranking when conflict is high but direct harm is lower.
- Penalize over-moderation when uncertainty and ambiguity are both high.
- Use current platform signals (visibility, engagement, reports, escalation) if provided.
- Return exactly one action from the allowed action space.

Output must be STRICT JSON with one of these shapes:
{"type":"tool","tool":"summarize_risk","input":{},"thought":"short string"}
{"type":"tool","tool":"inspect_action_space","input":{},"thought":"short string"}
{"type":"finish","decision":"downrank","thought":"short string","reasoning":"short string"}

Rules:
- JSON only.
- Keep thought/reasoning short.
- Never invent tools.
- Finish within 3 steps.
"""


@dataclass
class ActionDecision:
    action_id: int
    action_name: str
    reasoning: str
    trace: List[Dict[str, Any]]
    used_fallback: bool = False


def _as_state_dict(state: Any) -> Dict[str, Any]:
    if hasattr(state, "__dataclass_fields__"):
        return asdict(state)
    return dict(state)


def _clip01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def deterministic_action_from_state(
    state: Mapping[str, Any],
    runtime_signals: Optional[Mapping[str, Any]] = None,
) -> ActionDecision:
    """Safe fallback and baseline mapping from evaluation scores to actions."""
    runtime_signals = runtime_signals or {}

    harm = max(
        _clip01(state.get("toxicity")),
        _clip01(state.get("harassment")),
        _clip01(state.get("hate")),
        _clip01(state.get("self_harm")),
        _clip01(state.get("sexual")),
    )
    conflict = max(
        _clip01(state.get("conflict_risk")),
        _clip01(state.get("disagreement")),
        _clip01(state.get("escalation_level")),
    )
    uncertainty = _clip01(state.get("uncertainty"))
    ambiguity = _clip01(state.get("ambiguity"))
    engagement_risk = _clip01(state.get("engagement_risk"))
    reports = float(runtime_signals.get("reports", runtime_signals.get("R", 0.0)) or 0.0)
    escalation = _clip01(runtime_signals.get("escalation", runtime_signals.get("S", 0.0)))

    if harm >= 0.85 or (harm >= 0.70 and (reports >= 2 or escalation >= 0.65)):
        name = "throttle"
    elif harm >= 0.55 or conflict >= 0.75 or engagement_risk >= 0.70:
        name = "add_friction"
    elif harm >= 0.25 or conflict >= 0.35:
        name = "downrank"
    else:
        name = "do_nothing"

    if uncertainty >= 0.75 and ambiguity >= 0.75 and name in {"add_friction", "throttle"}:
        name = "downrank"

    return ActionDecision(
        action_id=ACTION_NAME_TO_ID[name],
        action_name=name,
        reasoning=(
            f"fallback policy chose {name} from harm={harm:.2f}, conflict={conflict:.2f}, "
            f"uncertainty={uncertainty:.2f}, ambiguity={ambiguity:.2f}, reports={reports:.1f}, escalation={escalation:.2f}"
        ),
        trace=[],
        used_fallback=True,
    )


class ReActActionChooser:
    def __init__(
        self,
        client: Any,
        model_name: str,
        max_steps: int = 3,
        max_api_retries: int = 3,
        retry_delay_seconds: float = 1.0,
    ) -> None:
        self._client = client
        self._model_name = model_name
        self._max_steps = max_steps
        self._max_api_retries = max(1, int(max_api_retries))
        self._retry_delay_seconds = max(0.0, float(retry_delay_seconds))

    def choose_action(
        self,
        state: Mapping[str, Any],
        runtime_signals: Optional[Mapping[str, Any]] = None,
    ) -> ActionDecision:
        state_dict = _as_state_dict(state)
        runtime_signals = dict(runtime_signals or {})
        trace: List[Dict[str, Any]] = []

        try:
            for step in range(self._max_steps):
                response = self._run_step(state_dict, runtime_signals, trace)
                trace.append({"step": step + 1, "model": response})

                if response.get("type") == "tool":
                    tool_name = response.get("tool")
                    observation = self._run_tool(tool_name, state_dict, runtime_signals)
                    trace.append(
                        {
                            "step": step + 1,
                            "tool": tool_name,
                            "observation": observation,
                        }
                    )
                    continue

                if response.get("type") == "finish":
                    decision_name = str(response.get("decision", "")).strip()
                    if decision_name not in ACTION_NAME_TO_ID:
                        raise ValueError(f"Unknown action '{decision_name}'")
                    return ActionDecision(
                        action_id=ACTION_NAME_TO_ID[decision_name],
                        action_name=decision_name,
                        reasoning=str(response.get("reasoning", "")).strip()
                        or str(response.get("thought", "")).strip(),
                        trace=trace,
                        used_fallback=False,
                    )

                raise ValueError(f"Unexpected response type: {response.get('type')}")
        except Exception as exc:
            fallback = deterministic_action_from_state(state_dict, runtime_signals)
            fallback.trace = trace + [{"error": str(exc)}]
            return fallback

        fallback = deterministic_action_from_state(state_dict, runtime_signals)
        fallback.trace = trace + [{"error": "max_steps_exceeded"}]
        return fallback

    def _run_step(
        self,
        state: Dict[str, Any],
        runtime_signals: Dict[str, Any],
        trace: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        payload = {
            "perception_state": state,
            "runtime_signals": runtime_signals,
            "available_actions": ACTION_DESCRIPTIONS,
            "scratchpad": trace,
        }

        last_exc: Exception | None = None
        for attempt in range(1, self._max_api_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                "Decide the next ReAct step for action selection.\n\n"
                                f"INPUT:\n{json.dumps(payload, ensure_ascii=False)}\n\n"
                                "Return JSON only."
                            ),
                        },
                    ],
                    temperature=0.1,
                )
                text = resp.choices[0].message.content
                return json.loads(text)
            except Exception as exc:
                last_exc = exc
                if attempt >= self._max_api_retries:
                    break
                time.sleep(self._retry_delay_seconds * attempt)

        assert last_exc is not None
        raise last_exc

    def _run_tool(
        self,
        tool_name: str,
        state: Mapping[str, Any],
        runtime_signals: Mapping[str, Any],
    ) -> Dict[str, Any]:
        if tool_name == "summarize_risk":
            return self._tool_summarize_risk(state, runtime_signals)
        if tool_name == "inspect_action_space":
            return self._tool_inspect_action_space(state, runtime_signals)
        raise ValueError(f"Unsupported tool '{tool_name}'")

    def _tool_summarize_risk(
        self,
        state: Mapping[str, Any],
        runtime_signals: Mapping[str, Any],
    ) -> Dict[str, Any]:
        harm_scores = {
            key: _clip01(state.get(key))
            for key in ["toxicity", "harassment", "hate", "self_harm", "sexual"]
        }
        conflict_scores = {
            key: _clip01(state.get(key))
            for key in ["conflict_risk", "disagreement", "escalation_level"]
        }
        top_harm = max(harm_scores, key=harm_scores.get)
        top_conflict = max(conflict_scores, key=conflict_scores.get)

        return {
            "max_harm": harm_scores[top_harm],
            "top_harm_dimension": top_harm,
            "max_conflict": conflict_scores[top_conflict],
            "top_conflict_dimension": top_conflict,
            "ambiguity": _clip01(state.get("ambiguity")),
            "uncertainty": _clip01(state.get("uncertainty")),
            "engagement_risk": _clip01(state.get("engagement_risk")),
            "reports": float(runtime_signals.get("reports", runtime_signals.get("R", 0.0)) or 0.0),
            "visibility": _clip01(runtime_signals.get("visibility", runtime_signals.get("V", 1.0))),
            "escalation": _clip01(runtime_signals.get("escalation", runtime_signals.get("S", 0.0))),
            "top_reasons": list(state.get("top_reasons") or [])[:3],
        }

    def _tool_inspect_action_space(
        self,
        state: Mapping[str, Any],
        runtime_signals: Mapping[str, Any],
    ) -> Dict[str, Any]:
        summary = self._tool_summarize_risk(state, runtime_signals)
        max_harm = summary["max_harm"]
        max_conflict = summary["max_conflict"]
        uncertainty = summary["uncertainty"]
        ambiguity = summary["ambiguity"]

        bands = {
            "do_nothing": "best when both harm and conflict are low",
            "downrank": "best when mild warning signs exist or model confidence is weak",
            "add_friction": "best when risk is meaningful but a full throttle may be too costly",
            "throttle": "best when severe harm or rapid escalation dominates false-positive cost",
        }

        default_choice = deterministic_action_from_state(state, runtime_signals).action_name
        caution = []
        if uncertainty >= 0.7:
            caution.append("high uncertainty suggests avoiding the strongest action unless harm is clearly severe")
        if ambiguity >= 0.7:
            caution.append("high ambiguity suggests preferring reversible interventions")
        if max_conflict >= 0.7 and max_harm < 0.5:
            caution.append("high conflict with lower direct harm often fits downrank or friction")
        if max_harm >= 0.8:
            caution.append("very high harm can justify throttle")

        return {
            "action_descriptions": ACTION_DESCRIPTIONS,
            "decision_bands": bands,
            "default_choice": default_choice,
            "caution_notes": caution,
        }


def build_react_action_chooser_from_env() -> ReActActionChooser:
    load_dotenv()
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "The 'openai' package is required for the ReAct chooser. Install it with `pip install openai`."
        ) from exc

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("MOD_AGENT_OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No OpenAI API key configured. Set OPENAI_API_KEY or MOD_AGENT_OPENAI_API_KEY."
        )

    model_name = os.getenv("OPENAI_MODEL", "api-gpt-oss-120b")
    base_url = os.getenv("OPENAI_BASE_URL", "https://tritonai-api.ucsd.edu")
    max_api_retries = int(os.getenv("ACTION_CHOOSER_MAX_API_RETRIES", "3"))
    retry_delay_seconds = float(os.getenv("ACTION_CHOOSER_RETRY_DELAY_SECONDS", "1.0"))
    client = OpenAI(api_key=api_key, base_url=base_url, max_retries=max_api_retries)
    return ReActActionChooser(
        client=client,
        model_name=model_name,
        max_api_retries=max_api_retries,
        retry_delay_seconds=retry_delay_seconds,
    )
