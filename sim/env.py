# sim/env.py
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from .data import Item


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


@dataclass(frozen=True)
class ActionParams:
    # visibility multiplier applied immediately
    vis_mult: float
    # multiplier on report probability / harmfulness
    harm_mult: float
    # penalty cost for using the action (user experience / ops cost)
    cost: float


# 0: do_nothing, 1: downrank, 2: friction, 3: throttle
ACTION_TABLE: Dict[int, ActionParams] = {
    0: ActionParams(vis_mult=1.00, harm_mult=1.00, cost=0.00),
    1: ActionParams(vis_mult=0.70, harm_mult=0.85, cost=0.05),
    2: ActionParams(vis_mult=0.55, harm_mult=0.60, cost=0.12),
    3: ActionParams(vis_mult=0.30, harm_mult=0.35, cost=0.25),
}
ACTION_STRENGTH = {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0}


class ModerationSimEnv:
    """
    Minimal MDP environment for moderation control.

    Episode = one content item (one Jigsaw comment).
    - LLM-extracted content features are fixed (harm/conf/uncertainty/ambiguity).
    - Platform dynamics evolve: visibility V, engagement E, reports R, escalation S.
    - Agent chooses intervention actions each step.
    """

    def __init__(
        self,
        items: List[Item],
        T: int = 10,
        seed: int = 0,
        # if you *want* balanced sampling you can set pos_frac != dataset ratio
        pos_frac: float | None = None,
    ):
        self.items = items
        self.T = T
        self.rng = np.random.default_rng(seed)

        # optional stratified buckets (based on dataset label toxic)
        self.pos_items = [it for it in items if int(it.labels.get("toxic", 0)) == 1]
        self.neg_items = [it for it in items if int(it.labels.get("toxic", 0)) == 0]
        self.pos_frac = pos_frac

        # --- dynamics hyperparameters (tunable) ---
        # engagement intensity
        self.base_E = 1.0
        self.a_conf = 8.0
        self.a_S = 4.0

        # report probability logits
        self.b_harm = 3.0
        self.b_logE = 0.7
        self.b_S = 1.2

        # escalation update
        self.c_conf = 2.0
        self.c_reports = 0.9
        self.c_action = 0.7

        # visibility recovery per step (models "cooldown wears off")
        self.v_recover = 0.05

        # --- reward weights ---
        self.w_good = 0.10
        self.w_reports = 1.00
        self.w_cost = 0.50
        self.w_harm_spread = 0.15  # penalize harmful diffusion even w/o reports

        # episode state
        self.t = 0
        self.item: Item | None = None
        self.harm = 0.0
        self.conf = 0.0
        self.ambiguity = 0.0
        self.uncertainty = 0.0
        self.V = 1.0
        self.E = 0.0
        self.R = 0.0
        self.S = 0.0

    # --------- feature extraction ----------
    def _extract_latents(self, state: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
        tox = float(state.get("toxicity", 0.0))
        har = float(state.get("harassment", 0.0))
        hate = float(state.get("hate", 0.0))
        sexual = float(state.get("sexual", 0.0))
        self_harm = float(state.get("self_harm", 0.0))

        conflict = float(state.get("conflict_risk", 0.0))
        disagree = float(state.get("disagreement", 0.0))
        esc0 = float(state.get("escalation_level", 0.0))

        ambiguity = float(state.get("ambiguity", 0.0))
        uncertainty = float(state.get("uncertainty", 0.0))

        harm = max(tox, har, hate, sexual, self_harm)
        conf = max(conflict, disagree, esc0)

        return clip01(harm), clip01(conf), clip01(ambiguity), clip01(uncertainty), clip01(esc0)

    def _obs(self) -> np.ndarray:
        # keep numbers bounded
        E_norm = math.log1p(self.E) / 5.0
        R_norm = math.log1p(self.R) / 5.0

        return np.array(
            [
                self.harm,
                self.conf,
                self.ambiguity,
                self.uncertainty,
                self.V,
                E_norm,
                R_norm,
                self.S,
            ],
            dtype=np.float32,
        )

    # --------- gym-like API ----------
    def reset(self) -> np.ndarray:
        self.t = 0

        if self.pos_frac is None:
            self.item = random.choice(self.items)
        else:
            # stratified draw
            if self.rng.random() < self.pos_frac and self.pos_items:
                self.item = random.choice(self.pos_items)
            else:
                self.item = random.choice(self.neg_items) if self.neg_items else random.choice(self.items)

        st = self.item.state
        self.harm, self.conf, self.ambiguity, self.uncertainty, esc0 = self._extract_latents(st)

        self.V = 1.0
        self.E = 0.0
        self.R = 0.0
        self.S = esc0

        return self._obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        a = int(action)
        params = ACTION_TABLE[a]

        # --- apply action effect on visibility instantly ---
        self.V = clip01(self.V * params.vis_mult)
        # slight recovery towards 1.0 each step
        self.V = clip01(self.V + self.v_recover)

        # --- engagement increment (depends on conf + escalation + visibility) ---
        lam = self.V * (self.base_E + self.a_conf * self.conf + self.a_S * self.S)
        lam = max(lam, 0.01)
        delta_E = int(self.rng.poisson(lam=lam))

        # --- reports increment (depends on harm + current engagement + escalation) ---
        logit = self.b_harm * self.harm + self.b_logE * math.log1p(self.E) + self.b_S * self.S
        p_report = sigmoid(logit) * params.harm_mult
        p_report = float(np.clip(p_report, 0.0, 1.0))
        delta_R = int(self.rng.binomial(n=max(delta_E, 0), p=p_report))

        # --- update E/R ---
        self.E += float(delta_E)
        self.R += float(delta_R)

        # --- escalation update ---
        drive = sigmoid(self.c_conf * self.conf + self.c_reports * delta_R - self.c_action * ACTION_STRENGTH[a])
        self.S = clip01(0.8 * self.S + 0.2 * drive)

        # --- reward ---
        good = delta_E * (1.0 - self.harm)  # "healthy" engagement
        harm_spread = self.harm * delta_E   # harmful diffusion even if no one reports
        reward = (
            self.w_good * good
            - self.w_reports * delta_R
            - self.w_harm_spread * harm_spread
            - self.w_cost * params.cost
        )

        self.t += 1
        done = self.t >= self.T

        info = {
            "id": self.item.id if self.item else None,
            "label_toxic": int(self.item.labels.get("toxic", 0)) if self.item else None,
            "harm": self.harm,
            "conf": self.conf,
            "V": self.V,
            "delta_E": delta_E,
            "delta_R": delta_R,
            "E": self.E,
            "R": self.R,
            "S": self.S,
        }
        return self._obs(), float(reward), done, info