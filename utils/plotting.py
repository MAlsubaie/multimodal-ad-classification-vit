import os
import matplotlib.pyplot as plt
import seaborn as sns


def plot_CM(cm, plots_dir, title="Confusion Matrix"):
    """
    Plot and save a confusion matrix as a heatmap.

    Args:
        cm (np.ndarray): Confusion matrix.
        plots_dir (str): Directory to save the plot.
        title (str): Title of the plot.
    """
    os.makedirs(plots_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)

    save_path = os.path.join(plots_dir, 'confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Confusion matrix saved to {save_path}")


def plot_metric(history, key, plots_dir, title=None, ylabel=None):
    """
    Helper function to plot and save a single metric over epochs.

    Args:
        history (dict): Training history dictionary.
        key (str): Metric name (e.g., 'acc', 'auc', 'precision', 'recall').
        plots_dir (str): Directory to save plots.
        title (str, optional): Plot title. Defaults to metric name.
        ylabel (str, optional): Label for y-axis. Defaults to metric name.
    """
    if f"val_{key}" not in history or key not in history:
        print(f"[WARN] Missing keys for metric '{key}' — skipping plot.")
        return

    os.makedirs(plots_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))

    plt.plot(history[key], label="Train", color='b')
    plt.plot(history[f"val_{key}"], label="Validation", color='g')

    plt.title(title or f"Model {key.capitalize()}")
    plt.xlabel("Epochs")
    plt.ylabel(ylabel or key.capitalize())
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(plots_dir, f"{key}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] {key.capitalize()} plot saved to {save_path}")


def plot_training_history(history, plots_dir):
    """
    Plot training and validation metrics over epochs.

    Args:
        history (dict): Dictionary containing training and validation metrics.
        plots_dir (str): Directory to save the plots.
    """
    metrics = ["acc", "auc", "precision", "recall"]
    for metric in metrics:
        plot_metric(history, metric, plots_dir)
