from statistics import mean

from sim.data import load_items
from sim.env import ModerationSimEnv
from sim.policies import rule_policy, always_do_nothing, always_throttle


def run_episode_on_item(env, policy_fn, item):
    obs = env.reset_to_item(item)

    total_reward = 0.0
    final_engagement = 0.0
    final_reports = 0.0
    final_escalation = 0.0

    for _ in range(env.max_steps):
        action = policy_fn(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        final_engagement = info["E"]
        final_reports = info["R"]
        final_escalation = info["S"]
        if done:
            break

    return {
        "total_reward": total_reward,
        "final_engagement": final_engagement,
        "final_reports": final_reports,
        "final_escalation": final_escalation,
    }


def evaluate_policy_on_all_items(items, policy_fn, policy_name, max_steps=30, seed=0):
    rewards = []
    engagements = []
    reports = []
    escalations = []

    for item in items:
        env = ModerationSimEnv(items, max_steps=max_steps, seed=seed, pos_frac=None)
        result = run_episode_on_item(env, policy_fn, item)

        rewards.append(result["total_reward"])
        engagements.append(result["final_engagement"])
        reports.append(result["final_reports"])
        escalations.append(result["final_escalation"])

    print(f"\n=== {policy_name} ===")
    print(f"Average total reward:      {mean(rewards):.3f}")
    print(f"Average final engagement:  {mean(engagements):.3f}")
    print(f"Average final reports:     {mean(reports):.3f}")
    print(f"Average final escalation:  {mean(escalations):.3f}")

    return {
        "policy_name": policy_name,
        "avg_reward": mean(rewards),
        "avg_engagement": mean(engagements),
        "avg_reports": mean(reports),
        "avg_escalation": mean(escalations),
        "all_rewards": rewards,
        "all_engagements": engagements,
        "all_reports": reports,
        "all_escalations": escalations,
    }


def main():
    items = load_items("jigsaw_perception_output.jsonl")

    results = []
    results.append(evaluate_policy_on_all_items(items, always_do_nothing, "Always Do Nothing"))
    results.append(evaluate_policy_on_all_items(items, rule_policy, "Rule Policy"))
    results.append(evaluate_policy_on_all_items(items, always_throttle, "Always Throttle"))

    print("\n=== Summary ===")
    results = sorted(results, key=lambda x: x["avg_reward"], reverse=True)
    for r in results:
        print(
            f"{r['policy_name']:20s} | "
            f"reward={r['avg_reward']:.3f} | "
            f"engagement={r['avg_engagement']:.3f} | "
            f"reports={r['avg_reports']:.3f} | "
            f"escalation={r['avg_escalation']:.3f}"
        )


if __name__ == "__main__":
    main()