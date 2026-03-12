# sim/rollout_demo.py
import numpy as np

from sim.data import load_items
from sim.env import ModerationSimEnv
from sim.policies import rule_policy, always_do_nothing, always_throttle


def run_episode(env, policy_fn, name: str, fixed_item=None):
    total_r = 0.0
    print(f"\n=== {name} ===")
    
    if fixed_item is None:
        obs = env.reset()
    else:
        obs = env.reset_to_item(fixed_item)
        
    for t in range(env.max_steps):
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
    item = items[0]
    env = ModerationSimEnv(items, max_steps=30, seed=0, pos_frac=0.5)  # set pos_frac=0.5 if you want balance

    run_episode(env, always_do_nothing, "Always Do Nothing", item)
    run_episode(env, rule_policy, "Rule Policy", item)
    run_episode(env, always_throttle, "Always Throttle", item)


if __name__ == "__main__":
    main()