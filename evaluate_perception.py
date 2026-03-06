"""
Evaluation script for comparing LLM perception scores against Jigsaw ground truth labels.

This script:
1. Loads the perception output JSONL file
2. Extracts perception scores and Jigsaw labels
3. Creates comparison plots (ROC curves, distributions, correlations, etc.)
4. Computes evaluation metrics (AUC, precision, recall, F1)
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


def load_perception_output(jsonl_path: Path) -> List[Dict]:
    """Load the JSONL output file."""
    data = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def extract_scores_and_labels(data: List[Dict]) -> Dict[str, np.ndarray]:
    """
    Extract perception scores and Jigsaw labels into arrays.

    Returns:
        {
            "labels": {
                "toxic": np.array([0, 1, 0, ...]),
                "severe_toxic": np.array([...]),
                ...
            },
            "scores": {
                "toxicity": np.array([0.02, 0.85, ...]),
                "harassment": np.array([...]),
                ...
            }
        }
    """
    labels = {
        "toxic": [],
        "severe_toxic": [],
        "obscene": [],
        "threat": [],
        "insult": [],
        "identity_hate": [],
    }
    scores = {
        "toxicity": [],
        "harassment": [],
        "hate": [],
        "self_harm": [],
        "sexual": [],
        "conflict_risk": [],
        "escalation_level": [],
        "ambiguity": [],
        "uncertainty": [],
        "disagreement": [],
        "engagement_risk": [],
    }

    for item in data:
        if item.get("labels") is None:
            continue  # Skip test split items without labels

        # Extract labels
        for key in labels.keys():
            labels[key].append(item["labels"].get(key, 0))

        # Extract perception scores
        state = item.get("state", {})
        for key in scores.keys():
            scores[key].append(state.get(key, 0.0))

    # Convert to numpy arrays
    labels = {k: np.array(v) for k, v in labels.items()}
    scores = {k: np.array(v) for k, v in scores.items()}

    return {"labels": labels, "scores": scores}


def plot_roc_curves(labels: Dict[str, np.ndarray], scores: Dict[str, np.ndarray], output_dir: Path):
    """Plot ROC curves for each label type vs relevant perception scores."""
    # Map Jigsaw labels to most relevant perception scores
    label_to_score = {
        "toxic": "toxicity",
        "severe_toxic": "toxicity",
        "obscene": "sexual",
        "threat": "harassment",
        "insult": "harassment",
        "identity_hate": "hate",
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (label_name, label_array) in enumerate(labels.items()):
        score_name = label_to_score.get(label_name, "toxicity")
        score_array = scores[score_name]

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(label_array, score_array)
        roc_auc = auc(fpr, tpr)

        # Plot
        ax = axes[idx]
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{label_name.replace('_', ' ').title()}\nvs {score_name.title()}")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved ROC curves to {output_dir / 'roc_curves.png'}")


def plot_score_distributions(labels: Dict[str, np.ndarray], scores: Dict[str, np.ndarray], output_dir: Path):
    """Plot distribution of perception scores for positive vs negative labels."""
    # Focus on toxicity vs toxic label
    toxic_labels = labels["toxic"]
    toxicity_scores = scores["toxicity"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram comparison
    ax = axes[0]
    ax.hist(
        toxicity_scores[toxic_labels == 0],
        bins=50,
        alpha=0.6,
        label="Non-toxic (label=0)",
        color="green",
        density=True,
    )
    ax.hist(
        toxicity_scores[toxic_labels == 1],
        bins=50,
        alpha=0.6,
        label="Toxic (label=1)",
        color="red",
        density=True,
    )
    ax.set_xlabel("Toxicity Score")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Toxicity Scores by Label")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Box plot comparison
    ax = axes[1]
    data_for_box = [
        toxicity_scores[toxic_labels == 0],
        toxicity_scores[toxic_labels == 1],
    ]
    bp = ax.boxplot(data_for_box, labels=["Non-toxic", "Toxic"], patch_artist=True)
    bp["boxes"][0].set_facecolor("lightgreen")
    bp["boxes"][1].set_facecolor("lightcoral")
    ax.set_ylabel("Toxicity Score")
    ax.set_title("Toxicity Score Distribution (Box Plot)")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "score_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved score distributions to {output_dir / 'score_distributions.png'}")


def plot_correlation_heatmap(labels: Dict[str, np.ndarray], scores: Dict[str, np.ndarray], output_dir: Path):
    """Plot correlation heatmap between perception scores and Jigsaw labels."""
    # Combine labels and scores into a single matrix
    all_data = {}
    for name, arr in labels.items():
        all_data[f"label_{name}"] = arr
    for name, arr in scores.items():
        all_data[f"score_{name}"] = arr

    # Create correlation matrix
    data_matrix = np.column_stack(list(all_data.values()))
    corr_matrix = np.corrcoef(data_matrix.T)
    labels_list = list(all_data.keys())

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        corr_matrix,
        xticklabels=labels_list,
        yticklabels=labels_list,
        annot=False,  # Too many cells, skip annotations
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Correlation Matrix: Perception Scores vs Jigsaw Labels", fontsize=14, pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved correlation heatmap to {output_dir / 'correlation_heatmap.png'}")


def plot_precision_recall_curves(labels: Dict[str, np.ndarray], scores: Dict[str, np.ndarray], output_dir: Path):
    """Plot precision-recall curves for each label type."""
    label_to_score = {
        "toxic": "toxicity",
        "severe_toxic": "toxicity",
        "obscene": "sexual",
        "threat": "harassment",
        "insult": "harassment",
        "identity_hate": "hate",
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (label_name, label_array) in enumerate(labels.items()):
        score_name = label_to_score.get(label_name, "toxicity")
        score_array = scores[score_name]

        # Compute precision-recall curve
        precision, recall, thresholds = precision_recall_curve(label_array, score_array)
        avg_precision = auc(recall, precision)

        # Plot
        ax = axes[idx]
        ax.plot(recall, precision, color="blue", lw=2, label=f"PR (AP = {avg_precision:.3f})")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{label_name.replace('_', ' ').title()}\nvs {score_name.title()}")
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved precision-recall curves to {output_dir / 'precision_recall_curves.png'}")


def plot_confusion_matrices(labels: Dict[str, np.ndarray], scores: Dict[str, np.ndarray], output_dir: Path):
    """Plot confusion matrices at different thresholds for toxicity prediction."""
    toxic_labels = labels["toxic"]
    toxicity_scores = scores["toxicity"]

    thresholds = [0.3, 0.5, 0.7]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, threshold in enumerate(thresholds):
        predictions = (toxicity_scores >= threshold).astype(int)
        cm = confusion_matrix(toxic_labels, predictions)

        ax = axes[idx]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Predicted Non-toxic", "Predicted Toxic"],
            yticklabels=["Actual Non-toxic", "Actual Toxic"],
            ax=ax,
        )
        ax.set_title(f"Confusion Matrix (Threshold = {threshold})")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")

        # Calculate metrics
        precision = precision_score(toxic_labels, predictions, zero_division=0)
        recall = recall_score(toxic_labels, predictions, zero_division=0)
        f1 = f1_score(toxic_labels, predictions, zero_division=0)
        ax.text(
            0.5,
            -0.15,
            f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}",
            transform=ax.transAxes,
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved confusion matrices to {output_dir / 'confusion_matrices.png'}")


def print_metrics_summary(labels: Dict[str, np.ndarray], scores: Dict[str, np.ndarray]):
    """Print summary metrics to console."""
    label_to_score = {
        "toxic": "toxicity",
        "severe_toxic": "toxicity",
        "obscene": "sexual",
        "threat": "harassment",
        "insult": "harassment",
        "identity_hate": "hate",
    }

    print("\n" + "=" * 80)
    print("EVALUATION METRICS SUMMARY")
    print("=" * 80)

    for label_name, label_array in labels.items():
        score_name = label_to_score.get(label_name, "toxicity")
        score_array = scores[score_name]

        # Compute metrics at threshold 0.5
        predictions = (score_array >= 0.5).astype(int)
        precision = precision_score(label_array, predictions, zero_division=0)
        recall = recall_score(label_array, predictions, zero_division=0)
        f1 = f1_score(label_array, predictions, zero_division=0)

        # Compute AUC
        fpr, tpr, _ = roc_curve(label_array, score_array)
        roc_auc = auc(fpr, tpr)

        print(f"\n{label_name.replace('_', ' ').title()} (vs {score_name}):")
        print(f"  AUC-ROC:     {roc_auc:.4f}")
        print(f"  Precision:   {precision:.4f}")
        print(f"  Recall:       {recall:.4f}")
        print(f"  F1 Score:     {f1:.4f}")
        print(f"  Positive samples: {label_array.sum()} / {len(label_array)} ({100 * label_array.mean():.2f}%)")

    print("\n" + "=" * 80)


def main():
    """Main evaluation pipeline."""
    # Setup paths
    project_root = Path(__file__).parent
    jsonl_path = project_root / "jigsaw_perception_output.jsonl"
    output_dir = project_root / "evaluation_plots"
    output_dir.mkdir(exist_ok=True)

    if not jsonl_path.exists():
        print(f"Error: {jsonl_path} not found. Run run_perception_on_jigsaw.py first.")
        sys.exit(1)

    print(f"Loading perception output from {jsonl_path}...")
    data = load_perception_output(jsonl_path)
    print(f"Loaded {len(data)} items")

    print("Extracting scores and labels...")
    extracted = extract_scores_and_labels(data)
    labels = extracted["labels"]
    scores = extracted["scores"]

    # Check we have labels (train split)
    if all(len(arr) == 0 for arr in labels.values()):
        print("Error: No labels found. Make sure you're using the 'train' split.")
        sys.exit(1)

    print(f"Found {len(labels['toxic'])} items with labels")

    # Generate plots
    print("\nGenerating evaluation plots...")
    plot_roc_curves(labels, scores, output_dir)
    plot_precision_recall_curves(labels, scores, output_dir)
    plot_score_distributions(labels, scores, output_dir)
    plot_correlation_heatmap(labels, scores, output_dir)
    plot_confusion_matrices(labels, scores, output_dir)

    # Print metrics
    print_metrics_summary(labels, scores)

    print(f"\n✓ All evaluation plots saved to {output_dir}/")
    print("  - roc_curves.png")
    print("  - precision_recall_curves.png")
    print("  - score_distributions.png")
    print("  - correlation_heatmap.png")
    print("  - confusion_matrices.png")


if __name__ == "__main__":
    main()
