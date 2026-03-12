import argparse
import csv
from pathlib import Path

import numpy as np

from sim.data import load_items
from sim.env import ModerationSimEnv
from sim.policies import LinearSoftmaxPolicy
from train_policy import evaluate_policy, train_reinforce


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", type=str, default="jigsaw_perception_output.jsonl")
    p.add_argument("--episodes", type=int, default=1800)
    p.add_argument("--eval-episodes", type=int, default=500)
    p.add_argument("--train-T", type=int, default=10)
    p.add_argument("--eval-T", type=int, default=30)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--num-seeds", type=int, default=8)
    p.add_argument("--gamma", type=float, default=0.98)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--baseline-momentum", type=float, default=0.95)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--w-good", type=float, default=0.18)
    p.add_argument("--w-reports", type=float, default=0.75)
    p.add_argument("--w-cost", type=float, default=0.8)
    p.add_argument("--w-harm-spread", type=float, default=0.12)
    p.add_argument("--w-benign-suppression", type=float, default=0.55)
    p.add_argument("--save-best", type=str, default="sim/learned_policy_best.npz")
    p.add_argument("--save-summary", type=str, default="policy_multiseed_summary.csv")
    return p.parse_args()


def build_env(items, T: int, seed: int, args) -> ModerationSimEnv:
    return ModerationSimEnv(
        items,
        T=T,
        seed=seed,
        w_good=args.w_good,
        w_reports=args.w_reports,
        w_cost=args.w_cost,
        w_harm_spread=args.w_harm_spread,
        w_benign_suppression=args.w_benign_suppression,
    )


def action_rates(items, policy: LinearSoftmaxPolicy, args, episodes: int = 400) -> tuple[float, float]:
    env = build_env(items, T=args.eval_T, seed=777, args=args)
    counts = np.zeros(4, dtype=np.int64)
    for _ in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            a = int(policy.policy_fn(greedy=True)(obs))
            counts[a] += 1
            obs, _, done, _ = env.step(a)
    total = counts.sum()
    intervention_rate = float((counts[1] + counts[2] + counts[3]) / total)
    throttle_rate = float(counts[3] / total)
    return intervention_rate, throttle_rate


def main():
    args = parse_args()
    items = load_items(args.jsonl)
    rows = []
    best = None

    for seed in range(args.seed_start, args.seed_start + args.num_seeds):
        policy = LinearSoftmaxPolicy(obs_dim=8, n_actions=4, seed=seed)
        train_env = build_env(items, T=args.train_T, seed=seed, args=args)
        train_reinforce(
            env=train_env,
            policy=policy,
            episodes=args.episodes,
            gamma=args.gamma,
            lr=args.lr,
            baseline_momentum=args.baseline_momentum,
            l2=args.l2,
            print_every=0,
        )

        eval_env = build_env(items, T=args.eval_T, seed=seed + 1000, args=args)
        m = evaluate_policy(eval_env, policy.policy_fn(greedy=True), episodes=args.eval_episodes)
        intervention_rate, throttle_rate = action_rates(items, policy, args)

        row = {
            "seed": seed,
            "mean_return": m["mean"],
            "std_return": m["std"],
            "intervention_rate": intervention_rate,
            "throttle_rate": throttle_rate,
        }
        rows.append(row)
        print(
            f"seed={seed:2d} "
            f"mean_return={m['mean']:+.3f} "
            f"std={m['std']:.3f} "
            f"intervene={intervention_rate:.3f} "
            f"throttle={throttle_rate:.3f}"
        )

        score = m["mean"]
        if best is None or score > best["score"]:
            best = {"score": score, "seed": seed, "policy": policy, "metrics": row}

    summary_path = Path(args.save_summary)
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "mean_return", "std_return", "intervention_rate", "throttle_rate"])
        w.writeheader()
        for row in rows:
            w.writerow(row)

    np.savez(
        args.save_best,
        W=best["policy"].W,
        b=best["policy"].b,
        seed=best["seed"],
        obs_dim=8,
        n_actions=4,
        episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        train_T=args.train_T,
        eval_T=args.eval_T,
        gamma=args.gamma,
        lr=args.lr,
        baseline_momentum=args.baseline_momentum,
        l2=args.l2,
        w_good=args.w_good,
        w_reports=args.w_reports,
        w_cost=args.w_cost,
        w_harm_spread=args.w_harm_spread,
        w_benign_suppression=args.w_benign_suppression,
        best_mean_return=best["metrics"]["mean_return"],
        best_std_return=best["metrics"]["std_return"],
        best_intervention_rate=best["metrics"]["intervention_rate"],
        best_throttle_rate=best["metrics"]["throttle_rate"],
    )

    print(f"\nSaved summary: {summary_path}")
    print(f"Saved best policy: {args.save_best}")
    print(f"Best seed: {best['seed']} with mean return {best['metrics']['mean_return']:+.3f}")


if __name__ == "__main__":
    main()
