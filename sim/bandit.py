import numpy as np


class LinUCBAgent:
    """
    Disjoint LinUCB contextual bandit with one linear model per action.

    For each action a:
      p_a(x) = theta_a^T x + alpha * sqrt(x^T A_a^{-1} x)
      theta_a = A_a^{-1} b_a
    """

    def __init__(
        self,
        n_actions: int,
        n_features: int,
        alpha: float = 0.8,
        l2_reg: float = 1.0,
        rng_seed: int = 0,
    ) -> None:
        if n_actions <= 0:
            raise ValueError("n_actions must be positive")
        if n_features <= 0:
            raise ValueError("n_features must be positive")

        self.n_actions = int(n_actions)
        self.n_features = int(n_features)
        self.alpha = float(alpha)
        self.l2_reg = float(l2_reg)
        self.rng = np.random.default_rng(rng_seed)

        self.A = np.stack([np.eye(self.n_features) * self.l2_reg for _ in range(self.n_actions)], axis=0)
        self.b = np.zeros((self.n_actions, self.n_features), dtype=np.float64)

    def _scores(self, x: np.ndarray, alpha: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.n_features:
            raise ValueError(f"Expected feature size {self.n_features}, got {x.shape[0]}")

        scores = np.zeros(self.n_actions, dtype=np.float64)
        for a in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            mean = float(theta @ x)
            bonus = float(np.sqrt(np.maximum(x @ A_inv @ x, 0.0)))
            scores[a] = mean + alpha * bonus
        return scores

    def select_action(self, x: np.ndarray) -> int:
        scores = self._scores(x, alpha=self.alpha)
        max_score = float(np.max(scores))
        # random tie-break keeps behavior stable and avoids action bias
        ties = np.flatnonzero(np.isclose(scores, max_score))
        return int(self.rng.choice(ties))

    def greedy_action(self, x: np.ndarray) -> int:
        scores = self._scores(x, alpha=0.0)
        max_score = float(np.max(scores))
        ties = np.flatnonzero(np.isclose(scores, max_score))
        return int(self.rng.choice(ties))

    def update(self, x: np.ndarray, action: int, reward: float) -> None:
        a = int(action)
        if not (0 <= a < self.n_actions):
            raise ValueError(f"action out of range: {a}")

        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.n_features:
            raise ValueError(f"Expected feature size {self.n_features}, got {x.shape[0]}")

        self.A[a] += np.outer(x, x)
        self.b[a] += float(reward) * x

