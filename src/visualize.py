import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List

def plot_loss_curves(histories: Dict[str, Dict[str, list]], save_path: str = None):
    """Plot Training vs Validation Learning Curves (2x2 Grid)"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for idx, (name, hist) in enumerate(histories.items()):
        ax = axes[idx]
        epochs_range = range(1, len(hist["train_loss"]) + 1)
        ax.plot(epochs_range, hist["train_loss"], "o-", label="Train Loss", color="#1f77b4", linewidth=2)
        ax.plot(epochs_range, hist["val_loss"], "s--", label="Val Loss", color="#ff7f0e", linewidth=2)
        ax.set_title(f"{name}: Loss Progression", fontweight="bold", fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Binary Cross Entropy")
        ax.legend(fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Visualization] Saved loss curves to {save_path}")
    plt.close()


def plot_confusion_matrices(test_results: Dict[str, Dict[str, Any]], save_path: str = None):
    """Plot Confusion Matrices (2x2 Grid)"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (name, metrics) in enumerate(test_results.items()):
        sns.heatmap(
            metrics["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
            ax=axes[idx],
            cbar=False,
            annot_kws={"size": 14, "fontweight": "bold"}
        )
        axes[idx].set_title(f"{name} Confusion Matrix", fontweight="bold", fontsize=12)
        axes[idx].set_xlabel("Predicted Label")
        axes[idx].set_ylabel("True Label")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Visualization] Saved confusion matrices to {save_path}")
    plt.close()


def plot_model_comparison(test_results: Dict[str, Dict[str, Any]], save_path: str = None):
    """Plot Comparative Model Performance Bar Chart"""
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(test_results.keys())
    accs = [test_results[n]["accuracy"] * 100 for n in names]
    f1s = [test_results[n]["f1_macro"] * 100 for n in names]

    x = np.arange(len(names))
    width = 0.35

    rects1 = ax.bar(x - width/2, accs, width, label="Accuracy (%)", color="#3498db")
    rects2 = ax.bar(x + width/2, f1s, width, label="Macro F1-Score (%)", color="#2ecc71")

    ax.set_title("Recurrent Architectures Benchmark (IMDb 25,000 Test Set)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontweight="bold", fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.7)

    for rects in [rects1, rects2]:
        for r in rects:
            h = r.get_height()
            ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Visualization] Saved model comparison chart to {save_path}")
    plt.close()


def plot_tuning_sensitivity(
    seq_lengths: List[int],
    seq_tuning_results: Dict[int, Dict[str, float]],
    hidden_dims: List[int],
    hidden_tuning_results: Dict[int, Dict[str, float]],
    save_path: str = None
):
    """Plot Sequence Length and Hidden Dimension Sensitivity Curves Side-by-Side"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Sequence Length Sensitivity
    axes[0].plot(seq_lengths, [seq_tuning_results[s]["accuracy"] for s in seq_lengths], "o-", label="Accuracy (%)", color="#1f77b4", linewidth=2)
    axes[0].plot(seq_lengths, [seq_tuning_results[s]["f1_macro"] for s in seq_lengths], "s--", label="Macro F1 (%)", color="#2ca02c", linewidth=2)
    axes[0].set_title("LSTM Sensitivity to Sequence Length (L)", fontweight="bold")
    axes[0].set_xlabel("Max Sequence Length (tokens)")
    axes[0].set_ylabel("Score (%)")
    axes[0].legend()
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Hidden Dimension Sensitivity
    axes[1].plot(hidden_dims, [hidden_tuning_results[h]["accuracy"] for h in hidden_dims], "o-", label="Accuracy (%)", color="#e74c3c", linewidth=2)
    axes[1].plot(hidden_dims, [hidden_tuning_results[h]["f1_macro"] for h in hidden_dims], "s--", label="Macro F1 (%)", color="#9b59b6", linewidth=2)
    axes[1].set_title("LSTM Sensitivity to Hidden Dimension (H)", fontweight="bold")
    axes[1].set_xlabel("Hidden State Dimension (H)")
    axes[1].set_ylabel("Score (%)")
    axes[1].legend()
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Visualization] Saved tuning sensitivity curves to {save_path}")
    plt.close()

