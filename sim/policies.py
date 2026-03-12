from dataclasses import dataclass
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


@dataclass
class LinearSoftmaxPolicy:
    obs_dim: int = 8
    n_actions: int = 4
    seed: int = 0
    init_scale: float = 0.01

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.W = self.rng.normal(0.0, self.init_scale, size=(self.n_actions, self.obs_dim)).astype(np.float32)
        self.b = np.zeros(self.n_actions, dtype=np.float32)

    def action_probs(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32)
        logits = self.W @ x + self.b
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)
        return probs.astype(np.float32)

    def sample_action(self, obs: np.ndarray) -> int:
        probs = self.action_probs(obs)
        return int(self.rng.choice(self.n_actions, p=probs))

    def greedy_action(self, obs: np.ndarray) -> int:
        probs = self.action_probs(obs)
        return int(np.argmax(probs))

    def update(self, obs: np.ndarray, action: int, advantage: float, lr: float, l2: float = 0.0) -> None:
        x = np.asarray(obs, dtype=np.float32)
        probs = self.action_probs(x)
        grad_logits = -probs
        grad_logits[int(action)] += 1.0
        self.W += lr * advantage * np.outer(grad_logits, x)
        self.b += lr * advantage * grad_logits
        if l2 > 0.0:
            self.W *= (1.0 - lr * l2)

    def policy_fn(self, greedy: bool = True):
        if greedy:
            return lambda obs: self.greedy_action(obs)
        return lambda obs: self.sample_action(obs)


def load_linear_policy(npz_path: str, seed: int = 0) -> LinearSoftmaxPolicy:
    data = np.load(npz_path)
    obs_dim = int(data["obs_dim"])
    n_actions = int(data["n_actions"])
    policy = LinearSoftmaxPolicy(obs_dim=obs_dim, n_actions=n_actions, seed=seed)
    policy.W = np.asarray(data["W"], dtype=np.float32)
    policy.b = np.asarray(data["b"], dtype=np.float32)
    return policy


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
