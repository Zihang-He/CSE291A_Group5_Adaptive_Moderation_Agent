import argparse
import csv
from pathlib import Path

import numpy as np

from sim.data import load_items
from sim.env import ModerationSimEnv
from sim.policies import always_do_nothing, always_throttle, load_linear_policy, rule_policy
from sim.proxy_data import load_proxy_items_from_jigsaw_zip


def evaluate(items, policy_fn, episodes: int, T: int, seed: int, env_kwargs: dict) -> dict:
    rng = np.random.default_rng(seed)
    returns = []
    end_E = []
    end_R = []
    action_counts = np.zeros(4, dtype=np.int64)
    benign_interventions = 0
    benign_total = 0
    toxic_strong = 0
    toxic_total = 0

    for ep in range(episodes):
        env = ModerationSimEnv(items, T=T, seed=int(rng.integers(1 << 31)), **env_kwargs)
        obs = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            a = int(policy_fn(obs))
            action_counts[a] += 1
            obs, r, done, info = env.step(a)
            ep_ret += float(r)
            harm = float(info["harm"])
            if harm < 0.25:
                benign_total += 1
                if a > 0:
                    benign_interventions += 1
            if harm > 0.6:
                toxic_total += 1
                if a >= 2:
                    toxic_strong += 1
        returns.append(ep_ret)
        end_E.append(float(info["E"]))
        end_R.append(float(info["R"]))

    total_actions = float(np.sum(action_counts))
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_final_engagement": float(np.mean(end_E)),
        "mean_final_reports": float(np.mean(end_R)),
        "intervention_rate": float((action_counts[1] + action_counts[2] + action_counts[3]) / total_actions),
        "throttle_rate": float(action_counts[3] / total_actions),
        "benign_intervention_rate": float(benign_interventions / benign_total) if benign_total > 0 else 0.0,
        "toxic_strong_action_rate": float(toxic_strong / toxic_total) if toxic_total > 0 else 0.0,
    }


def write_csv(path: Path, rows: list[dict]):
    fields = [
        "policy",
        "mean_return",
        "std_return",
        "mean_final_engagement",
        "mean_final_reports",
        "intervention_rate",
        "throttle_rate",
        "benign_intervention_rate",
        "toxic_strong_action_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", type=str, default="jigsaw_perception_output.jsonl")
    p.add_argument("--zip-path", type=str, default="jigsaw-toxic-comment-classification-challenge.zip")
    p.add_argument("--max-items", type=int, default=1000)
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--T", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--learned-policy-path", type=str, default=None)
    p.add_argument("--w-good", type=float, default=0.10)
    p.add_argument("--w-reports", type=float, default=1.00)
    p.add_argument("--w-cost", type=float, default=0.50)
    p.add_argument("--w-harm-spread", type=float, default=0.15)
    p.add_argument("--w-benign-suppression", type=float, default=0.30)
    p.add_argument("--output-csv", type=str, default="policy_eval_summary.csv")
    return p.parse_args()


def main():
    args = parse_args()
    if Path(args.jsonl).exists():
        items = load_items(args.jsonl)
        source = args.jsonl
    else:
        items = load_proxy_items_from_jigsaw_zip(args.zip_path, args.max_items)
        source = f"{args.zip_path} (proxy state from labels)"
    if len(items) > args.max_items:
        items = items[: args.max_items]

    env_kwargs = {
        "w_good": args.w_good,
        "w_reports": args.w_reports,
        "w_cost": args.w_cost,
        "w_harm_spread": args.w_harm_spread,
        "w_benign_suppression": args.w_benign_suppression,
    }

    policies = [
        ("Do Nothing", always_do_nothing),
        ("Rule Policy", rule_policy),
        ("Always Throttle", always_throttle),
    ]
    if args.learned_policy_path:
        learned = load_linear_policy(args.learned_policy_path, seed=args.seed)
        policies.append(("Learned Policy", learned.policy_fn(greedy=True)))

    rows = []
    for name, fn in policies:
        m = evaluate(items, fn, episodes=args.episodes, T=args.T, seed=args.seed + 7, env_kwargs=env_kwargs)
        row = {"policy": name, **m}
        rows.append(row)

    out = Path(args.output_csv)
    write_csv(out, rows)

    print(f"Data source: {source}")
    print(f"Items used: {len(items)}")
    print(f"Saved summary: {out}")
    for row in rows:
        print(
            f"{row['policy']:16s} "
            f"ret={row['mean_return']:+8.3f} "
            f"std={row['std_return']:7.3f} "
            f"E={row['mean_final_engagement']:7.2f} "
            f"R={row['mean_final_reports']:7.2f} "
            f"intervene={row['intervention_rate']:.3f} "
            f"throttle={row['throttle_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
