# sim/policies.py
import numpy as np

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