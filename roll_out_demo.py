# sim/rollout_demo.py
import numpy as np

from sim.data import load_items
from sim.env import ModerationSimEnv
from sim.policies import (
    always_do_nothing,
    always_throttle,
    make_react_policy,
    rule_policy,
)


def run_episode(env, policy_fn, name: str):
    obs = env.reset()
    total_r = 0.0
    print(f"\n=== {name} ===")
    for t in range(env.T):
        a = policy_fn(obs)
        obs, r, done, info = env.step(a)
        total_r += r
        print(
            f"t={t:02d} a={a} "
            f"harm={info['harm']:.2f} conf={info['conf']:.2f} "
            f"V={info['V']:.2f} dE={info['delta_E']:3d} dR={info['delta_R']:3d} "
            f"E={info['E']:.1f} R={info['R']:.1f} S={info['S']:.2f} r={r:+.3f}"
        )
        if done:
            break
    print(f"Total reward: {total_r:+.3f}")


def main():
    items = load_items("jigsaw_perception_output.jsonl")
    env = ModerationSimEnv(items, T=10, seed=0, pos_frac=None)  # set pos_frac=0.5 if you want balance

    run_episode(env, always_do_nothing, "Always Do Nothing")
    run_episode(env, rule_policy, "Rule Policy")
    try:
        react_policy = make_react_policy(env)
        run_episode(env, react_policy, "ReAct Action Chooser")
    except Exception as exc:
        print(f"\n[skip] ReAct Action Chooser unavailable: {exc}")
    run_episode(env, always_throttle, "Always Throttle")


if __name__ == "__main__":
    main()
