import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sim.data import Item, load_items
from sim.env import ModerationSimEnv
from sim.policies import always_do_nothing, always_throttle, load_linear_policy, rule_policy
from sim.proxy_data import load_proxy_items_from_jigsaw_zip


def run_episode_collect(item: Item, policy_fn, max_steps: int, seed: int, env_kwargs: dict | None = None):
    env = ModerationSimEnv([item], T=max_steps, seed=seed, **(env_kwargs or {}))
    obs = env.reset()

    reward_curve = []
    engagement_curve = []
    report_curve = []
    cumulative_reward = 0.0

    for _ in range(env.T):
        action = int(policy_fn(obs))
        obs, reward, done, info = env.step(action)
        cumulative_reward += float(reward)
        reward_curve.append(cumulative_reward)
        engagement_curve.append(float(info["E"]))
        report_curve.append(float(info["R"]))
        if done:
            break

    return reward_curve, engagement_curve, report_curve


def evaluate_policy(items, policy_fn, name, max_steps=30, seed=0, env_kwargs: dict | None = None):
    reward_all = []
    engagement_all = []
    reports_all = []

    for i, item in enumerate(items):
        r, e, rep = run_episode_collect(
            item=item,
            policy_fn=policy_fn,
            max_steps=max_steps,
            seed=seed + i,
            env_kwargs=env_kwargs,
        )
        reward_all.append(r)
        engagement_all.append(e)
        reports_all.append(rep)

    reward_all = np.asarray(reward_all, dtype=np.float32)
    engagement_all = np.asarray(engagement_all, dtype=np.float32)
    reports_all = np.asarray(reports_all, dtype=np.float32)

    return {
        "name": name,
        "reward": reward_all.mean(axis=0),
        "engagement": engagement_all.mean(axis=0),
        "reports": reports_all.mean(axis=0),
    }


def plot_results(results, output_path: str):
    plt.style.use("dark_background")
    t = np.arange(len(results[0]["reward"]))
    color_map = {
        "Do Nothing": "#00E5FF",
        "Rule Policy": "#FFD54F",
        "Always Throttle": "#FF6E6E",
        "Learned Policy": "#7CFF6B",
    }
    fallback = ["#B388FF", "#FFB74D", "#80CBC4", "#FF8A80", "#90CAF9"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    fig.patch.set_facecolor("black")

    for i, r in enumerate(results):
        c = color_map.get(r["name"], fallback[i % len(fallback)])
        axes[0].plot(t, r["reward"], label=r["name"], color=c, linewidth=2)
    axes[0].set_title("Average Cumulative Reward")
    axes[0].grid(alpha=0.3)

    for i, r in enumerate(results):
        c = color_map.get(r["name"], fallback[i % len(fallback)])
        axes[1].plot(t, r["engagement"], label=r["name"], color=c, linewidth=2)
    axes[1].set_title("Average Engagement")
    axes[1].grid(alpha=0.3)

    for i, r in enumerate(results):
        c = color_map.get(r["name"], fallback[i % len(fallback)])
        axes[2].plot(t, r["reports"], label=r["name"], color=c, linewidth=2)
    axes[2].set_title("Average Reports")
    axes[2].set_xlabel("Time Step")
    axes[2].grid(alpha=0.3)

    for ax in axes:
        ax.legend()
        ax.tick_params(colors="white")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", type=str, default="jigsaw_perception_output.jsonl")
    p.add_argument("--zip-path", type=str, default="jigsaw-toxic-comment-classification-challenge.zip")
    p.add_argument("--max-items", type=int, default=600)
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--w-good", type=float, default=0.10)
    p.add_argument("--w-reports", type=float, default=1.00)
    p.add_argument("--w-cost", type=float, default=0.50)
    p.add_argument("--w-harm-spread", type=float, default=0.15)
    p.add_argument("--w-benign-suppression", type=float, default=0.30)
    p.add_argument("--learned-policy-path", type=str, default=None)
    p.add_argument("--output", type=str, default="baseline_policy_curves.png")
    return p.parse_args()


def main():
    args = parse_args()

    if os.path.exists(args.jsonl):
        items = load_items(args.jsonl)
        data_source = args.jsonl
    else:
        items = load_proxy_items_from_jigsaw_zip(args.zip_path, max_items=args.max_items)
        data_source = f"{args.zip_path} (proxy state from labels)"

    if len(items) > args.max_items:
        items = items[: args.max_items]

    env_kwargs = {
        "w_good": args.w_good,
        "w_reports": args.w_reports,
        "w_cost": args.w_cost,
        "w_harm_spread": args.w_harm_spread,
        "w_benign_suppression": args.w_benign_suppression,
    }

    results = [
        evaluate_policy(
            items,
            always_do_nothing,
            "Do Nothing",
            max_steps=args.max_steps,
            seed=args.seed,
            env_kwargs=env_kwargs,
        ),
        evaluate_policy(
            items,
            rule_policy,
            "Rule Policy",
            max_steps=args.max_steps,
            seed=args.seed,
            env_kwargs=env_kwargs,
        ),
        evaluate_policy(
            items,
            always_throttle,
            "Always Throttle",
            max_steps=args.max_steps,
            seed=args.seed,
            env_kwargs=env_kwargs,
        ),
    ]
    if args.learned_policy_path:
        learned_policy = load_linear_policy(args.learned_policy_path, seed=args.seed)
        results.append(
            evaluate_policy(
                items,
                learned_policy.policy_fn(greedy=True),
                "Learned Policy",
                max_steps=args.max_steps,
                seed=args.seed,
                env_kwargs=env_kwargs,
            )
        )
    plot_results(results, args.output)

    print(f"Data source: {data_source}")
    print(f"Items used: {len(items)}")
    print(f"Plot saved to: {args.output}")
    for r in results:
        print(f"{r['name']}: final avg cumulative reward = {r['reward'][-1]:+.3f}")


if __name__ == "__main__":
    main()
