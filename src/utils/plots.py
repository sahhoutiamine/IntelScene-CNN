"""Plot helpers (training curves, confusion matrix)."""
import matplotlib.pyplot as plt
import seaborn as sns


def plot_history(history: dict, save_path=None, title: str = ""):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a, metric in zip(ax, ["loss", "accuracy"]):
        a.plot(history[metric], label=f"train {metric}")
        a.plot(history[f"val_{metric}"], label=f"val {metric}")
        a.set_xlabel("epoch")
        a.set_title(f"{title} {metric}".strip())
        a.legend()
        a.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    return fig


def plot_confusion_matrix(cm, class_names, save_path=None, normalize: bool = False, title: str = ""):
    data = cm.astype(float) / cm.sum(axis=1, keepdims=True) if normalize else cm
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(data, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title or ("Confusion matrix (normalised)" if normalize else "Confusion matrix"))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    return fig
