# sim/rollout_demo.py
import argparse

from sim.data import load_items
from sim.env import ModerationSimEnv
from sim.policies import (
    always_do_nothing,
    always_throttle,
    load_linear_policy,
    make_react_policy,
    rule_policy,
)


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", type=str, default="jigsaw_perception_output.jsonl")
    p.add_argument("--T", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pos-frac", type=float, default=None)
    p.add_argument("--policy-path", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    items = load_items(args.jsonl)
    env = ModerationSimEnv(items, T=args.T, seed=args.seed, pos_frac=args.pos_frac)

    run_episode(env, always_do_nothing, "Always Do Nothing")
    run_episode(env, rule_policy, "Rule Policy")
    try:
        react_policy = make_react_policy(env)
        run_episode(env, react_policy, "ReAct Action Chooser")
    except Exception as exc:
        print(f"\n[skip] ReAct Action Chooser unavailable: {exc}")
    run_episode(env, always_throttle, "Always Throttle")
    if args.policy_path:
        learned_policy = load_linear_policy(args.policy_path, seed=args.seed)
        run_episode(env, learned_policy.policy_fn(greedy=True), "Learned Policy (Greedy)")


if __name__ == "__main__":
    main()
