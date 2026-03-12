import argparse
from typing import Callable, Dict, List, Tuple

import numpy as np

from sim.data import load_items
from sim.env import ModerationSimEnv
from sim.policies import LinearSoftmaxPolicy, always_do_nothing, always_throttle, rule_policy


def run_episode(
    env: ModerationSimEnv,
    policy_fn: Callable[[np.ndarray], int],
    collect: bool = False,
) -> Tuple[float, List[np.ndarray], List[int], List[float]]:
    obs = env.reset()
    total_reward = 0.0
    obs_buf: List[np.ndarray] = []
    act_buf: List[int] = []
    rew_buf: List[float] = []

    for _ in range(env.T):
        action = int(policy_fn(obs))
        next_obs, reward, done, _ = env.step(action)
        if collect:
            obs_buf.append(obs.copy())
            act_buf.append(action)
            rew_buf.append(float(reward))
        total_reward += float(reward)
        obs = next_obs
        if done:
            break

    return total_reward, obs_buf, act_buf, rew_buf


def discounted_returns(rewards: List[float], gamma: float) -> np.ndarray:
    out = np.zeros(len(rewards), dtype=np.float32)
    g = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        g = rewards[i] + gamma * g
        out[i] = g
    return out


def evaluate_policy(
    env: ModerationSimEnv,
    policy_fn: Callable[[np.ndarray], int],
    episodes: int,
) -> Dict[str, float]:
    returns = []
    for _ in range(episodes):
        ep_return, _, _, _ = run_episode(env, policy_fn, collect=False)
        returns.append(ep_return)
    arr = np.array(returns, dtype=np.float32)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def train_reinforce(
    env: ModerationSimEnv,
    policy: LinearSoftmaxPolicy,
    episodes: int,
    gamma: float,
    lr: float,
    baseline_momentum: float,
    l2: float,
    print_every: int,
) -> List[float]:
    running_baseline = 0.0
    history: List[float] = []

    for ep in range(1, episodes + 1):
        ep_return, obs_buf, act_buf, rew_buf = run_episode(env, policy.sample_action, collect=True)
        history.append(ep_return)

        rets = discounted_returns(rew_buf, gamma)
        for obs, act, g in zip(obs_buf, act_buf, rets):
            running_baseline = baseline_momentum * running_baseline + (1.0 - baseline_momentum) * float(g)
            adv = float(g) - running_baseline
            policy.update(obs=obs, action=act, advantage=adv, lr=lr, l2=l2)

        if print_every > 0 and ep % print_every == 0:
            window = history[max(0, ep - print_every) : ep]
            print(
                f"episode={ep:5d} "
                f"avg_return(last_{len(window)})={np.mean(window):+.4f} "
                f"baseline={running_baseline:+.4f}"
            )

    return history


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", type=str, default="jigsaw_perception_output.jsonl")
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--eval-episodes", type=int, default=300)
    p.add_argument("--T", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.98)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--baseline-momentum", type=float, default=0.95)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pos-frac", type=float, default=None)
    p.add_argument("--print-every", type=int, default=100)
    p.add_argument("--save-path", type=str, default="sim/learned_policy.npz")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    items = load_items(args.jsonl)
    policy = LinearSoftmaxPolicy(obs_dim=8, n_actions=4, seed=args.seed)

    def make_env(eval_seed: int) -> ModerationSimEnv:
        return ModerationSimEnv(items, T=args.T, seed=eval_seed, pos_frac=args.pos_frac)

    print("== before training ==")
    for name, fn in [
        ("always_do_nothing", always_do_nothing),
        ("rule_policy", rule_policy),
        ("always_throttle", always_throttle),
        ("learned_policy_init_greedy", policy.policy_fn(greedy=True)),
    ]:
        m = evaluate_policy(make_env(eval_seed=args.seed + 100), fn, episodes=args.eval_episodes)
        print(f"{name:28s} mean={m['mean']:+.4f} std={m['std']:.4f}")

    print("\n== training (REINFORCE) ==")
    history = train_reinforce(
        env=make_env(eval_seed=args.seed),
        policy=policy,
        episodes=args.episodes,
        gamma=args.gamma,
        lr=args.lr,
        baseline_momentum=args.baseline_momentum,
        l2=args.l2,
        print_every=args.print_every,
    )

    print("\n== after training ==")
    for name, fn in [
        ("always_do_nothing", always_do_nothing),
        ("rule_policy", rule_policy),
        ("always_throttle", always_throttle),
        ("learned_policy_greedy", policy.policy_fn(greedy=True)),
        ("learned_policy_sample", policy.policy_fn(greedy=False)),
    ]:
        m = evaluate_policy(make_env(eval_seed=args.seed + 200), fn, episodes=args.eval_episodes)
        print(f"{name:28s} mean={m['mean']:+.4f} std={m['std']:.4f}")

    if args.save_path:
        np.savez(
            args.save_path,
            W=policy.W,
            b=policy.b,
            seed=args.seed,
            obs_dim=policy.obs_dim,
            n_actions=policy.n_actions,
            episodes=args.episodes,
            gamma=args.gamma,
            lr=args.lr,
            baseline_momentum=args.baseline_momentum,
            l2=args.l2,
            avg_return_last_100=float(np.mean(history[-100:])) if history else 0.0,
        )
        print(f"\nsaved policy params to: {args.save_path}")


if __name__ == "__main__":
    main()
