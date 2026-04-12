import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from torchmetrics.classification import Accuracy, AUROC, Precision, Recall

from models.model import create_model
from datasets.custom_dataset import CustomDataset
from config import enhanced_config
from utils.plotting import plot_CM


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Test BiFPN-ViT Model")
    parser.add_argument('--weights_path', type=str, default='./weights/best_model.pth', help='Path to model weights')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--img_size', type=int, nargs=4, default=(128, 128, 128, 1), help='Image dimensions')
    parser.add_argument('--test_csv', type=str, default='./df_test.csv', help='Path to test CSV file')
    parser.add_argument('--save_dir', type=str, default='./weights_finalized', help='Directory to save results')
    parser.add_argument('--plots_dir', type=str, default='./training_plots', help='Directory to save plots')
    return parser.parse_args()


def compute_metrics(outputs, labels, metrics, device):
    """Compute classification metrics for model predictions."""
    logits = outputs[0] if isinstance(outputs, tuple) else outputs
    preds = torch.argmax(logits, dim=1)

    acc = metrics['accuracy'](preds, labels)
    prec = torch.nan_to_num(metrics['precision'](preds, labels), nan=0.0)
    reca = torch.nan_to_num(metrics['recall'](preds, labels), nan=0.0)

    # Compute AUC only if there are at least two classes
    if len(torch.unique(labels)) > 1:
        auc_val = metrics['auc'](logits, labels)
    else:
        auc_val = torch.tensor(0.0, device=device)

    return acc.item(), prec.item(), reca.item(), auc_val.item()


def main():
    args = parse_args()

    # Load model
    model = create_model(enhanced_config)
    model.load_state_dict(torch.load(args.weights_path, map_location='cpu'))

    # Load data
    df_test = pd.read_csv(args.test_csv)
    test_dataset = CustomDataset(df_test, img_shape=args.img_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    model.eval()

    # Initialize metrics
    num_classes = 3
    metrics = {
        'accuracy': Accuracy(task="multiclass", num_classes=num_classes).to(device),
        'auc': AUROC(task="multiclass", num_classes=num_classes).to(device),
        'precision': Precision(task="multiclass", num_classes=num_classes).to(device),
        'recall': Recall(task="multiclass", num_classes=num_classes).to(device),
    }

    # Accumulators
    total_loss = 0.0
    total_acc = 0.0
    total_prec = 0.0
    total_reca = 0.0
    total_auc = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, biomarkers, labels in tqdm(test_loader, desc="Testing"):
            images, biomarkers, labels = images.to(device), biomarkers.to(device), labels.to(device)

            outputs = model(images, biomarkers)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = model.compute_loss(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            acc, prec, reca, auc_val = compute_metrics(outputs, labels, metrics, device)
            total_acc += acc
            total_prec += prec
            total_reca += reca
            total_auc += auc_val

    # Average metrics
    n_batches = len(test_loader)
    test_metrics = {
        "Loss": total_loss / n_batches,
        "Accuracy": total_acc / n_batches,
        "Precision": total_prec / n_batches,
        "Recall": total_reca / n_batches,
        "AUC": total_auc / n_batches,
    }

    # Print metrics
    print("\n=== Test Results ===")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)

    # Save confusion matrix plot
    if args.plots_dir:
        os.makedirs(args.plots_dir, exist_ok=True)
        plot_CM(cm, args.plots_dir)
        print(f"\nConfusion matrix saved to {args.plots_dir}")


if __name__ == "__main__":
    main()
