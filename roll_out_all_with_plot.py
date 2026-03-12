import numpy as np
import matplotlib.pyplot as plt

from sim.data import load_items
from sim.env import ModerationSimEnv
from sim.policies import rule_policy, always_do_nothing, always_throttle


def run_episode_collect(env, policy_fn, item):
    obs = env.reset_to_item(item)

    reward_curve = []
    engagement_curve = []
    report_curve = []

    cumulative_reward = 0.0

    for _ in range(env.max_steps):

        action = policy_fn(obs)
        obs, reward, done, info = env.step(action)

        cumulative_reward += reward

        reward_curve.append(cumulative_reward)
        engagement_curve.append(info["E"])
        report_curve.append(info["R"])

        if done:
            break

    return reward_curve, engagement_curve, report_curve


def evaluate_policy(items, policy_fn, name, max_steps=30):

    reward_all = []
    engagement_all = []
    reports_all = []

    for item in items:

        env = ModerationSimEnv(items, max_steps=max_steps, seed=0)

        r,e,rep = run_episode_collect(env, policy_fn, item)

        reward_all.append(r)
        engagement_all.append(e)
        reports_all.append(rep)

    reward_all = np.array(reward_all)
    engagement_all = np.array(engagement_all)
    reports_all = np.array(reports_all)

    return {
        "name": name,
        "reward": reward_all.mean(axis=0),
        "engagement": engagement_all.mean(axis=0),
        "reports": reports_all.mean(axis=0)
    }


def main():

    items = load_items("jigsaw_perception_output.jsonl")

    results = []

    results.append(evaluate_policy(items, always_do_nothing, "Do Nothing"))
    results.append(evaluate_policy(items, rule_policy, "Rule Policy"))
    results.append(evaluate_policy(items, always_throttle, "Always Throttle"))

    plot_results(results)


def plot_results(results):

    plt.style.use("dark_background")

    t = np.arange(len(results[0]["reward"]))

    colors = ["#00E5FF", "#FFD54F", "#FF6E6E"]

    fig, axes = plt.subplots(3,1, figsize=(10,7), sharex=True)

    fig.patch.set_facecolor("black")

    # reward
    for r,c in zip(results,colors):
        axes[0].plot(t, r["reward"], label=r["name"], color=c, linewidth=2)

    axes[0].set_title("Average Cumulative Reward")
    axes[0].grid(alpha=0.3)

    # engagement
    for r,c in zip(results,colors):
        axes[1].plot(t, r["engagement"], label=r["name"], color=c, linewidth=2)

    axes[1].set_title("Average Engagement")
    axes[1].grid(alpha=0.3)

    # reports
    for r,c in zip(results,colors):
        axes[2].plot(t, r["reports"], label=r["name"], color=c, linewidth=2)

    axes[2].set_title("Average Reports")
    axes[2].set_xlabel("Time Step")
    axes[2].grid(alpha=0.3)

    for ax in axes:
        ax.legend()
        ax.tick_params(colors="white")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()