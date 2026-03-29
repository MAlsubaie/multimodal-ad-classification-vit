import os
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy, AUROC, Precision, Recall

from models.model import create_model
from datasets.custom_dataset import CustomDataset
from config import enhanced_config
from utils.plotting import plot_CM, plot_training_history


# -----------------------------------------------------
# Argument Parser
# -----------------------------------------------------
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train BiFPN-ViT Model")

    parser.add_argument('--train_csv', type=str, default='./df_train.csv', help='Path to training CSV file')
    parser.add_argument('--val_csv', type=str, default='./df_val.csv', help='Path to validation CSV file')
    parser.add_argument('--test_csv', type=str, default='./df_test.csv', help='Path to test CSV file')
    parser.add_argument('--weights_dir', type=str, default='./weights_finalized', help='Directory to save model weights')
    parser.add_argument('--plots_dir', type=str, default='./training_plots', help='Directory to save plots')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and validation')
    parser.add_argument('--img_size', type=int, nargs=4, default=(128, 128, 128, 1), help='Image dimensions')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')

    return parser.parse_args()


# -----------------------------------------------------
# Metric Computation
# -----------------------------------------------------
def compute_metrics(outputs, labels, metrics, device):
    """Compute performance metrics for a batch."""
    logits = outputs[0] if isinstance(outputs, tuple) else outputs
    preds = torch.argmax(logits, dim=1)

    acc = metrics['accuracy'](preds, labels)
    prec = torch.nan_to_num(metrics['precision'](preds, labels), nan=0.0)
    reca = torch.nan_to_num(metrics['recall'](preds, labels), nan=0.0)

    if len(torch.unique(labels)) > 1:
        auc_val = metrics['auc'](logits, labels)
    else:
        auc_val = torch.tensor(0.0, device=device)

    return acc.item(), prec.item(), reca.item(), auc_val.item()


# -----------------------------------------------------
# Training Logic
# -----------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, metrics, device):
    """Run one full epoch of training."""
    model.train()
    epoch_loss, epoch_acc, epoch_prec, epoch_reca, epoch_auc = 0.0, 0.0, 0.0, 0.0, 0.0

    progress = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in progress:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        loss = model.compute_loss(logits, labels)
        loss.backward()
        optimizer.step()

        acc, prec, reca, auc_val = compute_metrics(outputs, labels, metrics, device)

        epoch_loss += loss.item()
        epoch_acc += acc
        epoch_prec += prec
        epoch_reca += reca
        epoch_auc += auc_val

        progress.set_postfix(loss=epoch_loss / (progress.n + 1), acc=epoch_acc / (progress.n + 1))

    n_batches = len(dataloader)
    return {
        "loss": epoch_loss / n_batches,
        "acc": epoch_acc / n_batches,
        "precision": epoch_prec / n_batches,
        "recall": epoch_reca / n_batches,
        "auc": epoch_auc / n_batches,
    }


def validate_one_epoch(model, dataloader, metrics, device):
    """Run one full epoch of validation."""
    model.eval()
    val_loss, val_acc, val_prec, val_reca, val_auc = 0.0, 0.0, 0.0, 0.0, 0.0

    progress = tqdm(dataloader, desc="Validation", leave=False)
    with torch.no_grad():
        for images, labels in progress:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = model.compute_loss(logits, labels)

            acc, prec, reca, auc_val = compute_metrics(outputs, labels, metrics, device)

            val_loss += loss.item()
            val_acc += acc
            val_prec += prec
            val_reca += reca
            val_auc += auc_val

            progress.set_postfix(loss=val_loss / (progress.n + 1), acc=val_acc / (progress.n + 1))

    n_batches = len(dataloader)
    return {
        "loss": val_loss / n_batches,
        "acc": val_acc / n_batches,
        "precision": val_prec / n_batches,
        "recall": val_reca / n_batches,
        "auc": val_auc / n_batches,
    }


# -----------------------------------------------------
# Checkpointing
# -----------------------------------------------------
def save_checkpoint(model, epoch, val_acc, save_path, best_val_acc):
    """Save model if validation accuracy improves."""
    if val_acc > best_val_acc:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"✔ Model saved at epoch {epoch+1} with val_acc: {val_acc:.4f}")
        return val_acc
    return best_val_acc


# -----------------------------------------------------
# Main Training Function
# -----------------------------------------------------
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    df_train = pd.read_csv(args.train_csv)
    df_val = pd.read_csv(args.val_csv)
    df_test = pd.read_csv(args.test_csv)

    train_dataset = CustomDataset(df_train, img_shape=args.img_size)
    val_dataset = CustomDataset(df_val, img_shape=args.img_size)
    test_dataset = CustomDataset(df_test, img_shape=args.img_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Create model
    model = create_model(enhanced_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # Metrics
    num_classes = 3
    metrics = {
        "accuracy": Accuracy(task="multiclass", num_classes=num_classes).to(device),
        "auc": AUROC(task="multiclass", num_classes=num_classes).to(device),
        "precision": Precision(task="multiclass", num_classes=num_classes).to(device),
        "recall": Recall(task="multiclass", num_classes=num_classes).to(device),
    }

    # Training history
    history = {k: [] for k in ["loss", "acc", "auc", "precision", "recall",
                               "val_loss", "val_acc", "val_auc", "val_precision", "val_recall"]}

    best_val_acc = 0.0

    # Training Loop
    for epoch in range(args.epochs):
        print(f"\n===== Epoch {epoch + 1}/{args.epochs} =====")

        train_metrics = train_one_epoch(model, train_loader, optimizer, metrics, device)
        val_metrics = validate_one_epoch(model, val_loader, metrics, device)

        for k in train_metrics:
            history[k].append(train_metrics[k])
            history[f"val_{k}"].append(val_metrics[k])

        best_val_acc = save_checkpoint(model, epoch, val_metrics["acc"],
                                       os.path.join(args.weights_dir, "best_model.pth"),
                                       best_val_acc)

        scheduler.step(val_metrics["loss"])

        print(f"Epoch {epoch + 1}/{args.epochs} Summary:")
        print(f"Train -> Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f}")
        print(f"Val   -> Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.4f}")

    # -------------------------------------------------
    # Final Testing
    # -------------------------------------------------
    model.eval()
    all_preds, all_labels = [], []
    test_loss, test_acc, test_prec, test_reca, test_auc = 0, 0, 0, 0, 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = model.compute_loss(logits, labels)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            acc, prec, reca, auc_val = compute_metrics(outputs, labels, metrics, device)
            test_loss += loss.item()
            test_acc += acc
            test_prec += prec
            test_reca += reca
            test_auc += auc_val

    n_batches = len(test_loader)
    print("\n===== Test Results =====")
    print(f"Loss: {test_loss / n_batches:.4f}")
    print(f"Accuracy: {test_acc / n_batches:.4f}")
    print(f"Precision: {test_prec / n_batches:.4f}")
    print(f"Recall: {test_reca / n_batches:.4f}")
    print(f"AUC: {test_auc / n_batches:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(all_labels, all_preds))

    # -------------------------------------------------
    # Save Plots
    # -------------------------------------------------
    os.makedirs(args.plots_dir, exist_ok=True)
    plot_CM(cm, args.plots_dir)
    plot_training_history(history, args.plots_dir)


if __name__ == "__main__":
    main()
