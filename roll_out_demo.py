# sim/rollout_demo.py
import argparse
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

from sim.bandit import LinUCBAgent
from sim.data import load_items
from sim.env import ModerationSimEnv
from sim.policies import always_do_nothing, always_throttle, rule_policy

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    sns = None

if sns is not None:
    sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


ACTION_NAME_TO_ID = {
    "do_nothing": 0,
    "downrank": 1,
    "add_friction": 2,
    "friction": 2,
    "throttle": 3,
}


def suggested_action_one_hot(item) -> np.ndarray:
    """
    Encode LLM prior action into a 4-dim one-hot vector.
    If missing/unknown, return all zeros.
    """
    vec = np.zeros(4, dtype=np.float64)
    if item is None:
        return vec

    action_name = str(item.state.get("suggested_action", "")).strip().lower()
    action_id = ACTION_NAME_TO_ID.get(action_name)
    if action_id is not None:
        vec[action_id] = 1.0
    return vec


def featurize(obs: np.ndarray, prior_one_hot: np.ndarray) -> np.ndarray:
    """Context for LinUCB: env obs + suggested_action one-hot + bias."""
    return np.concatenate(
        [
            obs.astype(np.float64),
            prior_one_hot.astype(np.float64),
            np.array([1.0], dtype=np.float64),
        ],
        axis=0,
    )


def train_linucb(
    items,
    train_episodes: int = 2000,
    horizon: int = 10,
    alpha: float = 0.8,
    l2_reg: float = 1.0,
    seed: int = 0,
):
    env = ModerationSimEnv(items, T=horizon, seed=seed, pos_frac=None)
    obs0 = env.reset()
    prior0 = suggested_action_one_hot(env.item)
    n_features = featurize(obs0, prior0).shape[0]

    agent = LinUCBAgent(
        n_actions=4,
        n_features=n_features,
        alpha=alpha,
        l2_reg=l2_reg,
        rng_seed=seed,
    )

    ep_returns = []
    for ep in range(train_episodes):
        obs = env.reset()
        prior = suggested_action_one_hot(env.item)
        total_r = 0.0
        for _ in range(horizon):
            x = featurize(obs, prior)
            action = agent.select_action(x)
            next_obs, reward, done, _ = env.step(action)
            agent.update(x, action, reward)
            total_r += reward
            obs = next_obs
            if done:
                break
        ep_returns.append(total_r)

        if (ep + 1) % 400 == 0:
            recent = float(np.mean(ep_returns[-200:]))
            print(f"[train] ep={ep + 1:4d} avg_return(last 200)={recent:+.3f}")

    return agent, np.array(ep_returns, dtype=np.float64)


def evaluate_policy(items, policy_fn, episodes: int = 500, horizon: int = 10, seed: int = 123):
    env = ModerationSimEnv(items, T=horizon, seed=seed, pos_frac=None)
    returns = []
    for _ in range(episodes):
        obs = env.reset()
        total_r = 0.0
        for _ in range(horizon):
            action = int(policy_fn(obs, env))
            obs, reward, done, _ = env.step(action)
            total_r += reward
            if done:
                break
        returns.append(total_r)
    return float(np.mean(returns)), float(np.std(returns))


def run_one_verbose_episode(items, policy_fn, name: str, horizon: int = 10, seed: int = 999):
    env = ModerationSimEnv(items, T=horizon, seed=seed, pos_frac=None)
    obs = env.reset()
    total_r = 0.0

    print(f"\n=== {name} (single episode trace) ===")
    for t in range(horizon):
        action = int(policy_fn(obs, env))
        obs, reward, done, info = env.step(action)
        total_r += reward
        print(
            f"t={t:02d} a={action} "
            f"harm={info['harm']:.2f} conf={info['conf']:.2f} "
            f"V={info['V']:.2f} dE={info['delta_E']:3d} dR={info['delta_R']:3d} "
            f"E={info['E']:.1f} R={info['R']:.1f} S={info['S']:.2f} r={reward:+.3f}"
        )
        if done:
            break
    print(f"Total reward: {total_r:+.3f}")


def collect_stepwise_eval_data(items, policy_fn, episodes: int, horizon: int, seed: int = 123):
    """
    Collect per-step evaluation samples for plotting.

    score = intervention strength in [0, 1], where 0=no action, 1=throttle.
    """
    env = ModerationSimEnv(items, T=horizon, seed=seed, pos_frac=None)
    labels = []
    scores = []
    harms = []
    confs = []
    rewards = []

    for _ in range(episodes):
        obs = env.reset()
        for _ in range(horizon):
            action = int(policy_fn(obs, env))
            obs, reward, done, info = env.step(action)
            labels.append(int(info["label_toxic"]))
            scores.append(float(action) / 3.0)
            harms.append(float(info["harm"]))
            confs.append(float(info["conf"]))
            rewards.append(float(reward))
            if done:
                break

    return {
        "label_toxic": np.array(labels, dtype=np.int32),
        "action_score": np.array(scores, dtype=np.float64),
        "harm": np.array(harms, dtype=np.float64),
        "conf": np.array(confs, dtype=np.float64),
        "reward": np.array(rewards, dtype=np.float64),
    }


def draw_heatmap(ax, matrix: np.ndarray, xlabels, ylabels, title: str, fmt: str = ".2f"):
    if sns is not None:
        sns.heatmap(
            matrix,
            xticklabels=xlabels,
            yticklabels=ylabels,
            annot=True,
            fmt=fmt,
            cmap="coolwarm",
            center=0.0 if fmt == ".2f" else None,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )
    else:
        im = ax.imshow(matrix, cmap="coolwarm")
        ax.set_xticks(np.arange(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(ylabels)))
        ax.set_yticklabels(ylabels)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center", color="black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)


def plot_roc_curves(labels: np.ndarray, scores: np.ndarray, output_dir: Path):
    if len(np.unique(labels)) < 2:
        print("Skip ROC: labels have a single class.")
        return

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"LinUCB action score (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC: Toxic Label vs LinUCB Action Strength")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC curves to {output_dir / 'roc_curves.png'}")


def plot_precision_recall_curves(labels: np.ndarray, scores: np.ndarray, output_dir: Path):
    if len(np.unique(labels)) < 2:
        print("Skip PR: labels have a single class.")
        return

    precision, recall, _ = precision_recall_curve(labels, scores)
    ap = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="blue", lw=2, label=f"LinUCB action score (AP = {ap:.3f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR Curve: Toxic Label vs LinUCB Action Strength")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved precision-recall curves to {output_dir / 'precision_recall_curves.png'}")


def plot_score_distributions(labels: np.ndarray, scores: np.ndarray, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(
        scores[labels == 0],
        bins=20,
        alpha=0.6,
        label="Non-toxic",
        color="green",
        density=True,
    )
    ax.hist(
        scores[labels == 1],
        bins=20,
        alpha=0.6,
        label="Toxic",
        color="red",
        density=True,
    )
    ax.set_xlabel("LinUCB Action Strength (a/3)")
    ax.set_ylabel("Density")
    ax.set_title("Action Score Distribution by Toxic Label")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    box_data = [scores[labels == 0], scores[labels == 1]]
    bp = ax.boxplot(box_data, labels=["Non-toxic", "Toxic"], patch_artist=True)
    bp["boxes"][0].set_facecolor("lightgreen")
    bp["boxes"][1].set_facecolor("lightcoral")
    ax.set_ylabel("LinUCB Action Strength (a/3)")
    ax.set_title("Action Score Distribution (Box Plot)")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "score_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved score distributions to {output_dir / 'score_distributions.png'}")


def plot_correlation_heatmap(eval_data: dict, output_dir: Path):
    keys = ["label_toxic", "action_score", "harm", "conf", "reward"]
    data_matrix = np.column_stack([eval_data[k] for k in keys])
    with np.errstate(divide="ignore", invalid="ignore"):
        corr_matrix = np.corrcoef(data_matrix.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    fig, ax = plt.subplots(figsize=(8, 6))
    draw_heatmap(ax, corr_matrix, keys, keys, "Correlation Heatmap: Policy Signals", fmt=".2f")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved correlation heatmap to {output_dir / 'correlation_heatmap.png'}")


def plot_confusion_matrices(labels: np.ndarray, scores: np.ndarray, output_dir: Path):
    thresholds = [0.3, 0.5, 0.7]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, threshold in enumerate(thresholds):
        preds = (scores >= threshold).astype(int)
        cm = confusion_matrix(labels, preds, labels=[0, 1])

        ax = axes[idx]
        draw_heatmap(
            ax,
            cm,
            ["Pred Non-toxic", "Pred Toxic"],
            ["Actual Non-toxic", "Actual Toxic"],
            f"Threshold = {threshold}",
            fmt="d",
        )
        ax.set_ylabel("True Label")
        ax.set_xlabel("Pred Label")

        p = precision_score(labels, preds, zero_division=0)
        r = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        ax.text(
            0.5,
            -0.20,
            f"Precision: {p:.3f}, Recall: {r:.3f}, F1: {f1:.3f}",
            transform=ax.transAxes,
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrices to {output_dir / 'confusion_matrices.png'}")


def generate_policy_plots(eval_data: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = eval_data["label_toxic"]
    scores = eval_data["action_score"]

    plot_roc_curves(labels, scores, output_dir)
    plot_precision_recall_curves(labels, scores, output_dir)
    plot_score_distributions(labels, scores, output_dir)
    plot_correlation_heatmap(eval_data, output_dir)
    plot_confusion_matrices(labels, scores, output_dir)

    print(f"\nAll policy plots saved to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate a LinUCB moderation policy.")
    parser.add_argument("--data", type=str, default="jigsaw_perception_output.jsonl")
    parser.add_argument("--train-episodes", type=int, default=2000)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--plot-dir", type=str, default="policy_evaluation_plots")
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. "
            "Run `python run_perception_on_jigsaw.py` first to generate it."
        )

    items = load_items(str(data_path))

    train_episodes = args.train_episodes
    eval_episodes = args.eval_episodes
    horizon = args.horizon

    print("Training LinUCB contextual bandit...")
    agent, train_returns = train_linucb(
        items,
        train_episodes=train_episodes,
        horizon=horizon,
        alpha=0.8,
        l2_reg=1.0,
        seed=0,
    )
    print(f"[train] mean_return(all)={float(np.mean(train_returns)):+.3f}")

    def linucb_greedy(obs: np.ndarray, env: ModerationSimEnv) -> int:
        prior = suggested_action_one_hot(env.item)
        return agent.greedy_action(featurize(obs, prior))

    baselines = [
        ("Always Do Nothing", lambda obs, _env: always_do_nothing(obs)),
        ("Rule Policy", lambda obs, _env: rule_policy(obs)),
        ("Always Throttle", lambda obs, _env: always_throttle(obs)),
        ("LinUCB (Greedy)", linucb_greedy),
    ]

    print("\n=== Evaluation (mean episode return) ===")
    for name, fn in baselines:
        mean_r, std_r = evaluate_policy(items, fn, episodes=eval_episodes, horizon=horizon, seed=123)
        print(f"{name:20s}  mean={mean_r:+.3f}  std={std_r:.3f}")

    print("\nGenerating policy evaluation plots...")
    eval_data = collect_stepwise_eval_data(
        items,
        linucb_greedy,
        episodes=eval_episodes,
        horizon=horizon,
        seed=123,
    )
    generate_policy_plots(eval_data, Path(args.plot_dir))

    run_one_verbose_episode(items, linucb_greedy, "LinUCB (Greedy)", horizon=horizon, seed=999)


if __name__ == "__main__":
    main()
