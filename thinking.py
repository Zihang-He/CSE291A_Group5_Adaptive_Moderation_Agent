"""
Thinking moderation policy

For each step, the agent will:
    - Constructs a thinking prompt from the current observation and memory.
    - Calls the LLM to 'think' about the situation and propose an action.
    - Parses the structured output and returns an integer action.

Usage
    from thinking import ThinkingPolicy
    policy = ThinkingPolicy()
    obs = env.reset()
    for t in range(env.T):
        action = policy(obs)
        obs, r, done, info = env.step(action)
        policy.record_step(action, r, info)
"""
import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI

# The same order as the env's obnservation.
OBS_FIELDS = ["harm", "conf", "ambiguity", "uncertainty", "V", "E_norm", "R_norm", "S"]

# We have four actions: 0 = do_nothing, 1 = downrank, 2 = add_friction, 3 = throttle.
ACTION_NAMES = {0: "do_nothing", 1: "downrank", 2: "add_friction", 3: "throttle"}

T = 10  # episode length, for prompt construction

SEED = 42  # random seed. Don't ask why it is 42 :).

@dataclass
class Step:
    """One step of episode memory."""
    t: int
    obs_dict: Dict[str, float]
    action: int
    reward: float
    info: Dict[str, Any]

SYSTEM_PROMPT = """\
You are a content moderation engine inside a 
simulated social media platform. Your job is to think about the current 
state of a piece of content and then choose the best moderation action.

Each episode tracks ONE content item over T=10 time steps.
At each step the platform state evolves:
    - V  : current visibility (0‒1).  Actions reduce it; it recovers for every step.
    - E  : engagement count (views / interactions).
    - R  : user reports.
    - S  : escalation level (0‒1), rises with conflict and reports.

Static content features (extracted by an LLM perception module):
    - harm        : max(toxicity, harassment, hate, sexual, self_harm) in [0,1]
    - conf        : max(conflict_risk, disagreement, escalation_level) in [0,1]
    - ambiguity   : how ambiguous the content is in [0,1]
    - uncertainty : model uncertainty about its own perception in [0,1]

Available Actions:
    0: do_nothing   : full visibility, no cost
    1: downrank     : mild reduction (vis×0.70, cost 0.05)
    2: add_friction : moderate (vis×0.55, cost 0.12)
    3: throttle     : aggressive (vis×0.30, cost 0.25)

Reward:
    - 0.10 × healthy_engagement
    − 1.00 × reports
    − 0.15 × harmful_spread
    − 0.50 × action_cost

Goal: maximize cumulative reward over the episode.
    - For non-harmful content: let it pass and earn engagement reward.
    - For harmful content: suppress early to prevent reports and harm spread.
    - Avoid over-moderating non-harmful content.
    - Track escalation: if S is climbing, you may need to escalate your action.

Output Format: Respond in this structure (do not generate other format):

<think>
[You reasoning. Which action you chose and why in 2‒5 sentences.]
</think>
<action>[0, 1, 2, or 3]</action>
"""

def obs_to_dict(obs: np.ndarray) -> Dict[str, float]:
    """Convert obs array into a dict with field names."""
    return {name: round(float(obs[i]), 4) for i, name in enumerate(OBS_FIELDS)}

def build_user_prompt(
    obs_dict: Dict[str, float],
    history: List[Step],
    cumulative_reward: float,
    t: int,
) -> str:
    """Construct the user prompt for the LLM, including current observation and recent history."""
    lines: List[str] = []

    lines.append(f"Step {t} / 9")
    lines.append("Current observation:")
    for k, v in obs_dict.items():
        lines.append(f" {k} = {v}")
    lines.append(f" cumulative_Reward = {cumulative_reward}")

    if history:
        lines.append("\nRecent history:")
        # Only show the last 5 steps to avoid overwhelming the prompt. Recent steps are more relevant, so we reverse the order.
        top_histories = history[-5:][::-1]
        for recent in top_histories:
            a_name = ACTION_NAMES[recent.action]
            lines.append(
                f"  t={recent.t}  action={a_name}  "
                f"dE={recent.info.get('delta_E', '?')}  dR={recent.info.get('delta_R', '?')}  "
                f"S={recent.info.get('S', 0):.2f}  reward={recent.reward:+.3f}"
            )

    if obs_dict.get("harm", 0) < 0.1 and obs_dict.get("conf", 0) < 0.15:
        lines.append("\nThis content is clearly not harmful. Try not to do the moderation on it.")
    if obs_dict.get("harm", 0) > 0.9 and obs_dict.get("conf", 0) > 0.8:
        lines.append("\nThis content is likely harmful and risky. Consider taking stronger moderation actions early to prevent harm spread and user reports.")
    
    lines.append("\nChoose your action.")
    return "\n".join(lines)


def parse_action(text: str) -> int:
    """Extract the action integer from the LLM response text."""
    m = re.search(r"<action>\s*(\d)\s*</action>", text)
    if m and int(m.group(1)) in range(4):
        return int(m.group(1))
    digits = re.findall(r"\b([0-3])\b", text[-100:])
    return int(digits[-1]) if digits else 0


def parse_thinking(text: str) -> str:
    """Extract the thinking text from the LLM response."""
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def get_client() -> tuple[AsyncOpenAI, str]:
    """Build AsyncOpenAI client from env vars."""
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

    base_url = os.getenv("OPENAI_BASE_URL", "https://tritonai-api.ucsd.edu")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    return client, model_name


class ThinkingPolicy:
    """
    Drop-in policy that uses an LLM call to decide moderation actions.

    Params:
        client : AsyncOpenAI, default None
            Pre-built client.  If None, reads env vars.
        model : str, default None
            Model name.  If None, reads OPENAI_MODEL env var.
        temperature, default 0.2
            Sampling temperature.
        verbose, default True
            Print thinking traces.
    """

    def __init__(
        self,
        temperature: float = 0.2,
        verbose: bool = True,
    ):
        self.client, self.model = get_client()
        self.temperature = temperature
        self.verbose = verbose

        self.history: List[Step] = []
        self.cum_reward: float = 0.0
        self.t: int = 0
        self.last_thinking: str = ""

    async def call_llm(self, user_msg: str) -> str:
        """Make the async API call to get the LLM response."""
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=self.temperature,
        )
        return resp.choices[0].message.content

    def __call__(self, obs: np.ndarray) -> int:
        """Synchronous wrapper around the async LLM call."""
        obs_dict = obs_to_dict(obs)
        user_msg = build_user_prompt(
            obs_dict, self.history, self.cum_reward, self.t
        )

        try:
            text = asyncio.run(self.call_llm(user_msg))
        except RuntimeError:
            # already inside an event loop — use nest_asyncio or thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                text = pool.submit(asyncio.run, self.call_llm(user_msg)).result()
        except Exception as e:
            if self.verbose:
                print(f"  [ThinkingPolicy] API error: {e} → fallback")
            from sim.policies import rule_policy
            return rule_policy(obs)

        action = parse_action(text)
        self.last_thinking = parse_thinking(text)

        if self.verbose:
            print(f"  [Think] {self.last_thinking}")
            print(f"  [Act]   {action} ({ACTION_NAMES[action]})")

        return action

    async def act_async(self, obs: np.ndarray) -> int:
        """Async version of act, for use in async environments."""
        obs_dict = obs_to_dict(obs)
        user_msg = build_user_prompt(
            obs_dict, self.history, self.cum_reward, self.t
        )
        try:
            text = await self.call_llm(user_msg)
        except Exception as e:
            if self.verbose:
                print(f"  [ThinkingPolicy] API error: {e} → fallback")
            from sim.policies import rule_policy
            return rule_policy(obs)

        action = parse_action(text)
        self.last_thinking = parse_thinking(text)

        if self.verbose:
            print(f"  [Think] {self.last_thinking}")
            print(f"  [Act]   {action} ({ACTION_NAMES[action]})")

        return action

    def record_step(self, action: int, reward: float, info: Dict[str, Any]):
        self.history.append(Step(
            t=self.t, obs_dict={}, action=action, reward=reward, info=info,
        ))
        self.cum_reward += reward
        self.t += 1

    def reset(self):
        self.history.clear()
        self.cum_reward = 0.0
        self.t = 0
        self.last_thinking = ""

def run_llm_episode(env, policy: ThinkingPolicy, label: str = "Thinking Policy"):
    policy.reset()
    obs = env.reset()
    total_r = 0.0
    item = env.item
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Content: {item.id}  |  Toxic: {item.labels.get('toxic', '?')}")
    print(f"  harm={env.harm:.2f}  conf={env.conf:.2f}")
    print(f"{'='*60}")
    for t in range(env.T):
        print(f"\n--- t={t} ---")
        action = policy(obs)
        obs, r, done, info = env.step(action)
        policy.record_step(action, r, info)
        total_r += r
        print(f"Result: dE={info['delta_E']:3d}  dR={info['delta_R']:3d}  "
              f"V={info['V']:.2f}  S={info['S']:.2f}  reward={r:+.3f}")
        if done:
            break
    print(f"\nTotal episode reward: {total_r:+.3f}")
    return total_r


if __name__ == "__main__":
    from sim.data import load_items
    from sim.env import ModerationSimEnv
    items = load_items("jigsaw_perception_output.jsonl")
    env = ModerationSimEnv(items, T=T, seed=SEED, pos_frac=0.5)
    policy = ThinkingPolicy(verbose=True)
    for ep in range(3):
        run_llm_episode(env, policy, f"LLM Episode {ep+1}")