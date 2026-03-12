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
        self, items: List[Item], 
        max_steps: int = 10, 
        seed: int = 0, 
        # balanced sampling set pos_frac != dataset ratio
        pos_frac: float | None = None
    ):

        self.items = items
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

        self.pos_items = [it for it in items if int(it.labels.get("toxic", 0)) == 1]
        self.neg_items = [it for it in items if int(it.labels.get("toxic", 0)) == 0]
        self.pos_frac = pos_frac

        # engagement dynamics parameters
        self.base_engagement_rate = 1.0
        self.conflict_boost = 8.0
        self.escalation_boost = 4.0

        # report probability parameters
        self.harm_weight = 3.0
        self.engagement_weight = 0.7
        self.escalation_weight = 1.2

        # escalation update parameters
        self.conflict_drive = 2.0
        self.report_drive = 0.9
        self.action_suppression = 0.7

        # visibility recovery
        self.visibility_recovery = 0.05

        # reward weights
        self.weight_healthy_engagement = 0.20
        self.weight_reports = 1.00
        self.weight_intervention_cost = 0.50
        self.weight_harmful_spread = 0.30
        self.weight_benign_suppression = 0.25

        # episode state
        self.current_step = 0
        self.item: Item | None = None

        self.harm_score = 0.0
        self.conflict_score = 0.0
        self.ambiguity = 0.0
        self.uncertainty = 0.0

        self.visibility = 1.0
        self.total_engagement = 0.0
        self.total_reports = 0.0
        self.escalation_level = 0.0
        
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
        E_norm = math.log1p(self.total_engagement) / 5.0
        R_norm = math.log1p(self.total_reports) / 5.0

        return np.array(
            [
                self.harm_score,
                self.conflict_score,
                self.ambiguity,
                self.uncertainty,
                self.visibility,
                E_norm,
                R_norm,
                self.escalation_level,
            ],
            dtype=np.float32,
        )

    # --------- gym-like API ----------
    def reset(self) -> np.ndarray:
        self.current_step = 0

        if self.pos_frac is None:
            self.item = random.choice(self.items)
        else:
            # stratified draw
            if self.rng.random() < self.pos_frac and self.pos_items:
                self.item = random.choice(self.pos_items)
            else:
                self.item = random.choice(self.neg_items) if self.neg_items else random.choice(self.items)

        st = self.item.state
        self.harm_score, self.conflict_score, self.ambiguity, self.uncertainty, self.escalation_level = self._extract_latents(st)

        self.visibility = 1.0
        self.total_engagement = 0.0
        self.total_reports = 0.0
        self.escalation_level = 0.0

        return self._obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        a = int(action)
        params = ACTION_TABLE[a]

        # --- apply action effect on visibility instantly ---
        self.visibility = clip01(self.visibility * params.vis_mult)
        # slight recovery towards 1.0 each step
        self.visibility = clip01(self.visibility + self.visibility_recovery)

        # --- engagement increment (depends on conf + escalation + visibility) ---
        lam = self.visibility * (self.base_engagement_rate + self.conflict_boost * self.conflict_score + self.escalation_boost * self.escalation_level)
        lam = max(lam, 0.01)
        delta_E = int(self.rng.poisson(lam=lam))

        # --- reports increment (depends on harm + current engagement + escalation) ---
        logit = self.harm_weight * self.harm_score + self.engagement_weight * math.log1p(self.total_engagement) + self.escalation_weight * self.escalation_level
        p_report = sigmoid(logit) * params.harm_mult
        p_report = float(np.clip(p_report, 0.0, 1.0))
        delta_R = int(self.rng.binomial(n=max(delta_E, 0), p=p_report))

        # --- update E/R ---
        self.total_engagement += float(delta_E)
        self.total_reports += float(delta_R)

        # --- escalation update ---
        drive = sigmoid(self.conflict_drive * self.conflict_score + self.report_drive * delta_R - self.action_suppression * ACTION_STRENGTH[a])
        self.escalation_level = clip01(0.8 * self.escalation_level + 0.2 * drive)

        # --- reward ---
        good = delta_E * (1.0 - self.harm_score)  # "healthy" engagement
        harm_spread = self.harm_score * delta_E   # harmful diffusion even if no one reports
        
        benign_suppression_penalty = (1 - self.harm_score) * ACTION_STRENGTH[a]
        reward = (
            self.weight_healthy_engagement * good
            - self.weight_reports * delta_R
            - self.weight_harmful_spread * harm_spread
            - self.weight_intervention_cost * params.cost
            - self.weight_benign_suppression * benign_suppression_penalty
        )

        self.current_step += 1
        done = self.current_step >= self.max_steps

        info = {
            "id": self.item.id if self.item else None,
            "label_toxic": int(self.item.labels.get("toxic", 0)) if self.item else None,
            "harm": self.harm_score,
            "conf": self.conflict_score,
            "V": self.visibility,
            "delta_E": delta_E,
            "delta_R": delta_R,
            "E": self.total_engagement,
            "R": self.total_reports,
            "S": self.escalation_level,
        }
        return self._obs(), float(reward), done, info