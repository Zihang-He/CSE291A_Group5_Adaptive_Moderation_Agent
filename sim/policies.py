import numpy as np

from action_chooser import (
    ReActActionChooser,
    build_react_action_chooser_from_env,
    deterministic_action_from_state,
)

# obs = [harm, conf, ambiguity, uncertainty, V, logE, logR, S]

def always_do_nothing(obs: np.ndarray) -> int:
    return 0

def always_throttle(obs: np.ndarray) -> int:
    return 3

def rule_policy(obs: np.ndarray) -> int:
    harm = float(obs[0])
    conf = float(obs[1])

    # treat "conflict" slightly differently: high-conf but low-harm => friction/downrank
    if harm > 0.8:
        return 3  # throttle
    if harm > 0.5:
        return 2  # friction
    if conf > 0.7:
        return 2  # friction (de-escalate)
    if harm > 0.25 or conf > 0.35:
        return 1  # downrank
    return 0


def make_react_policy(env, chooser: ReActActionChooser | None = None):
    """
    Build a policy callable that selects actions from the current env item state
    using the ReAct chooser. Falls back to a deterministic policy if the model
    is unavailable or returns invalid output.
    """
    chooser = chooser or build_react_action_chooser_from_env()

    def _policy(obs: np.ndarray) -> int:
        if env.item is None:
            raise RuntimeError("Environment item is not initialized. Call env.reset() first.")

        runtime_signals = {
            "visibility": float(obs[4]),
            "engagement": float(obs[5]),
            "reports": float(obs[6]),
            "escalation": float(obs[7]),
            "V": float(env.V),
            "E": float(env.E),
            "R": float(env.R),
            "S": float(env.S),
        }
        decision = chooser.choose_action(env.item.state, runtime_signals=runtime_signals)
        return decision.action_id

    return _policy


def react_fallback_policy_from_state(state: dict, runtime_signals: dict | None = None) -> int:
    return deterministic_action_from_state(state, runtime_signals).action_id
