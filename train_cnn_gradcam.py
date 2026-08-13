"""
Train a custom CNN binary classifier and generate Grad-CAM explanations.

This script uses only balanced split CSVs, filters missing/corrupted images,
saves reusable filtered CSVs, trains a compact CNN, evaluates on test data, and
exports Grad-CAM visualisations for TP/TN/FP/FN examples.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import cv2
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps, UnidentifiedImageError
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN and generate Grad-CAM explanations.")
    parser.add_argument("--train_csv", required=True, type=Path)
    parser.add_argument("--val_csv", required=True, type=Path)
    parser.add_argument("--test_csv", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--image_size", default=224, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--limit_train", default=None, type=int)
    parser.add_argument("--limit_val", default=None, type=int)
    parser.add_argument("--limit_test", default=None, type=int)
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    return device


def ensure_balanced_csv_name(path: Path, expected_name: str) -> None:
    if path.name.lower() != expected_name:
        raise ValueError(f"Expected {expected_name}, got {path}. Use balanced CSVs only.")


def stratified_limit(df: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    if limit <= 0:
        raise ValueError("Limit values must be positive.")
    parts = []
    remaining = limit
    labels = sorted(df["label"].unique().tolist())
    for i, label in enumerate(labels):
        label_df = df[df["label"] == label]
        if i == len(labels) - 1:
            take = min(len(label_df), remaining)
        else:
            take = min(len(label_df), max(1, int(round(limit * len(label_df) / len(df)))))
        parts.append(label_df.sample(n=take, random_state=seed))
        remaining -= take
    return pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed).head(limit).reset_index(drop=True)


def is_readable_image(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (OSError, UnidentifiedImageError, ValueError):
        return False


def read_and_filter_csv(
    csv_path: Path,
    expected_name: str,
    split_name: str,
    output_dir: Path,
    limit: int | None,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    ensure_balanced_csv_name(csv_path, expected_name)
    df = pd.read_csv(csv_path)
    required = {"sample_path", "label"}
    missing_columns = required.difference(df.columns)
    if missing_columns:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing_columns)}")

    df["label"] = df["label"].astype(int)
    unexpected = set(df["label"].unique()).difference({0, 1})
    if unexpected:
        raise ValueError(f"{csv_path} contains labels outside {{0, 1}}: {sorted(unexpected)}")

    original_rows = len(df)
    if limit is not None and limit < len(df):
        df = stratified_limit(df, limit, seed)

    before = len(df)
    exists_mask = df["sample_path"].astype(str).map(lambda p: Path(p).exists())
    missing_removed = int((~exists_mask).sum())
    df = df[exists_mask].copy()

    readable_mask = df["sample_path"].astype(str).map(is_readable_image)
    corrupted_removed = int((~readable_mask).sum())
    df = df[readable_mask].copy().reset_index(drop=True)

    filtered_path = output_dir / "filtered_csvs" / f"{split_name}_filtered.csv"
    df.to_csv(filtered_path, index=False, encoding="utf-8-sig")

    stats = {
        "original_rows": original_rows,
        "checked_rows": before,
        "missing_removed": missing_removed,
        "corrupted_removed": corrupted_removed,
        "after": len(df),
    }
    return df, stats


def print_class_counts(split_name: str, df: pd.DataFrame, stats: dict[str, int]) -> None:
    counts = df["label"].value_counts().to_dict()
    print(
        f"{split_name}: Real={counts.get(0, 0)} | Manipulated={counts.get(1, 0)} | "
        f"Total={len(df)} | missing removed={stats['missing_removed']} | "
        f"corrupted removed={stats['corrupted_removed']}"
    )


class FaceManipulationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: transforms.Compose | None = None) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        path = str(row["sample_path"])
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {
            "image": img,
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "sample_path": path,
        }


class SimpleCNN(nn.Module):
    def __init__(self, dropout: float = 0.4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, 32),
            self._block(32, 64),
            self._block(64, 128),
            self._block(128, 256),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(256, 1)

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    @property
    def final_conv_layer(self) -> nn.Module:
        return self.features[-1][0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x).squeeze(1)


def make_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tfms, eval_tfms


def worker_init_fn(worker_id: int) -> None:
    np.random.seed(torch.initial_seed() % 2**32)
    random.seed(torch.initial_seed() % 2**32)


def make_loader(
    df: pd.DataFrame,
    transform: transforms.Compose,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        FaceManipulationDataset(df, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
        generator=generator,
    )


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=["Real", "Manipulated"],
            output_dict=True,
            zero_division=0,
        ),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    progress = tqdm(loader, desc=f"Train epoch {epoch}", unit="batch", dynamic_ncols=True)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        progress.set_postfix(loss=total_loss / max(total_samples, 1))
    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str,
) -> tuple[float, np.ndarray, np.ndarray, list[str]]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_paths: list[str] = []

    for batch in tqdm(loader, desc=desc, unit="batch", dynamic_ncols=True):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        all_probs.append(probs.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
        all_paths.extend(batch["sample_path"])

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels).astype(int)
    return total_loss / max(total_samples, 1), y_true, y_prob, all_paths


def save_curves(history_df: pd.DataFrame, plots_dir: Path) -> None:
    plt.figure()
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "cnn_loss_curve.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(history_df["epoch"], history_df["val_f1"], label="Val F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title("CNN Validation F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "cnn_f1_curve.png", dpi=200)
    plt.close()


def save_test_plots(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, plots_dir: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Manipulated"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("CNN Confusion Matrix")
    plt.tight_layout()
    plt.savefig(plots_dir / "cnn_confusion_matrix.png", dpi=200)
    plt.close()

    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_value = roc_auc_score(y_true, y_prob)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC-AUC = {auc_value:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("CNN ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(plots_dir / "cnn_roc_curve.png", dpi=200)
        plt.close()
    except ValueError:
        print("ROC curve skipped because only one class is present.")


class GradCAM:
    def __init__(self, model: SimpleCNN, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module: nn.Module, inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _save_gradient(
        self,
        module: nn.Module,
        grad_input: tuple[torch.Tensor],
        grad_output: tuple[torch.Tensor],
    ) -> None:
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(self, image_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor.unsqueeze(0).to(device))
        logits[0].backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze(0)
        cam = torch.relu(cam)
        cam_np = cam.detach().cpu().numpy()
        cam_np = cv2.resize(cam_np, (image_tensor.shape[2], image_tensor.shape[1]))
        cam_np = cam_np - cam_np.min()
        denom = cam_np.max()
        if denom > 0:
            cam_np = cam_np / denom
        return cam_np


def load_original_rgb(path: str, image_size: int) -> np.ndarray:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img = img.resize((image_size, image_size), Image.Resampling.BILINEAR)
    return np.asarray(img)


def tensor_for_gradcam(path: str, image_size: int) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
    return transform(img)


def save_gradcam_visual(
    original_rgb: np.ndarray,
    heatmap: np.ndarray,
    output_path: Path,
) -> None:
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.55 * original_rgb + 0.45 * heatmap_color)

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for ax in axes:
        ax.axis("off")
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original")
    axes[1].imshow(heatmap_color)
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def generate_gradcams(
    model: SimpleCNN,
    predictions_df: pd.DataFrame,
    output_dir: Path,
    image_size: int,
    device: torch.device,
) -> pd.DataFrame:
    gradcam_root = output_dir / "gradcam"
    selections = {
        "TP": predictions_df[(predictions_df.true_label == 1) & (predictions_df.predicted_label == 1)].head(5),
        "TN": predictions_df[(predictions_df.true_label == 0) & (predictions_df.predicted_label == 0)].head(5),
        "FP": predictions_df[(predictions_df.true_label == 0) & (predictions_df.predicted_label == 1)].head(5),
        "FN": predictions_df[(predictions_df.true_label == 1) & (predictions_df.predicted_label == 0)].head(5),
    }

    gradcam = GradCAM(model, model.final_conv_layer)
    records: list[dict[str, Any]] = []
    try:
        for case_type, case_df in selections.items():
            case_dir = gradcam_root / case_type
            case_dir.mkdir(parents=True, exist_ok=True)
            for i, row in tqdm(
                enumerate(case_df.itertuples(index=False)),
                total=len(case_df),
                desc=f"Grad-CAM {case_type}",
                dynamic_ncols=True,
            ):
                try:
                    image_tensor = tensor_for_gradcam(row.sample_path, image_size)
                    heatmap = gradcam.generate(image_tensor, device)
                    original_rgb = load_original_rgb(row.sample_path, image_size)
                    output_path = case_dir / f"{case_type}_{i:03d}.png"
                    save_gradcam_visual(original_rgb, heatmap, output_path)
                    records.append(
                        {
                            "sample_path": row.sample_path,
                            "true_label": int(row.true_label),
                            "predicted_label": int(row.predicted_label),
                            "probability": float(row.probability),
                            "case_type": case_type,
                            "output_image_path": str(output_path.resolve()),
                        }
                    )
                except (OSError, UnidentifiedImageError, ValueError, RuntimeError) as exc:
                    print(f"Grad-CAM failed for {row.sample_path}: {exc}")
    finally:
        gradcam.remove_hooks()

    gradcam_index = pd.DataFrame(records)
    gradcam_index.to_csv(gradcam_root / "gradcam_index.csv", index=False, encoding="utf-8-sig")
    return gradcam_index


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    output_dir = args.output_dir.expanduser().resolve()
    models_dir = output_dir / "models"
    metrics_dir = output_dir / "metrics"
    plots_dir = output_dir / "plots"
    gradcam_dir = output_dir / "gradcam"
    filtered_dir = output_dir / "filtered_csvs"
    for folder in [models_dir, metrics_dir, plots_dir, gradcam_dir, filtered_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    train_df, train_stats = read_and_filter_csv(
        args.train_csv.expanduser().resolve(),
        "train_balanced.csv",
        "train",
        output_dir,
        args.limit_train,
        args.seed,
    )
    val_df, val_stats = read_and_filter_csv(
        args.val_csv.expanduser().resolve(),
        "val_balanced.csv",
        "val",
        output_dir,
        args.limit_val,
        args.seed,
    )
    test_df, test_stats = read_and_filter_csv(
        args.test_csv.expanduser().resolve(),
        "test_balanced.csv",
        "test",
        output_dir,
        args.limit_test,
        args.seed,
    )

    print("Class counts after filtering")
    print_class_counts("train", train_df, train_stats)
    print_class_counts("val", val_df, val_stats)
    print_class_counts("test", test_df, test_stats)

    train_tfms, eval_tfms = make_transforms(args.image_size)
    train_loader = make_loader(train_df, train_tfms, args.batch_size, True, args.num_workers, args.seed)
    val_loader = make_loader(val_df, eval_tfms, args.batch_size, False, args.num_workers, args.seed)
    test_loader = make_loader(test_df, eval_tfms, args.batch_size, False, args.num_workers, args.seed)

    model = SimpleCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    best_model_path = models_dir / "cnn_best_model.pt"
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_true, val_prob, _ = evaluate(model, val_loader, criterion, device, f"Validate epoch {epoch}")
        val_metrics = compute_binary_metrics(val_true, val_prob)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics["roc_auc"],
        }
        history.append(row)
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"val_f1={val_metrics['f1']:.4f}, val_auc={val_metrics['roc_auc']}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_f1": best_f1,
                    "args": serializable_args(args),
                },
                best_model_path,
            )
            print(f"Saved best model: {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    history_df = pd.DataFrame(history)
    history_df.to_csv(metrics_dir / "cnn_training_history.csv", index=False, encoding="utf-8-sig")
    with (metrics_dir / "cnn_training_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    save_curves(history_df, plots_dir)

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, y_true, y_prob, sample_paths = evaluate(model, test_loader, criterion, device, "Testing")
    y_pred = (y_prob >= 0.5).astype(int)
    test_metrics = compute_binary_metrics(y_true, y_prob)
    test_metrics["test_loss"] = float(test_loss)
    test_metrics["best_epoch"] = int(best_epoch)
    test_metrics["best_validation_f1"] = float(best_f1)

    predictions_df = pd.DataFrame(
        {
            "sample_path": sample_paths,
            "true_label": y_true.astype(int),
            "probability": y_prob.astype(float),
            "predicted_label": y_pred.astype(int),
        }
    )
    predictions_df.to_csv(metrics_dir / "cnn_test_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([test_metrics]).to_csv(metrics_dir / "cnn_test_metrics.csv", index=False, encoding="utf-8-sig")
    with (metrics_dir / "cnn_test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)
    save_test_plots(y_true, y_pred, y_prob, plots_dir)

    # Save a joblib pointer with useful metadata for downstream notebooks.
    joblib.dump(
        {
            "checkpoint": str(best_model_path.resolve()),
            "model_class": "SimpleCNN",
            "image_size": args.image_size,
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
        },
        models_dir / "cnn_model_metadata.joblib",
    )

    gradcam_index = generate_gradcams(model, predictions_df, output_dir, args.image_size, device)

    summary = {
        "best_validation_f1": float(best_f1),
        "best_epoch": int(best_epoch),
        "test_metrics": test_metrics,
        "filtering": {
            "train": train_stats,
            "val": val_stats,
            "test": test_stats,
        },
        "saved_model": str(best_model_path.resolve()),
        "gradcam_index": str((gradcam_dir / "gradcam_index.csv").resolve()),
        "gradcam_count": int(len(gradcam_index)),
    }
    with (metrics_dir / "cnn_run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nCNN + Grad-CAM complete")
    print(f"Best validation F1: {best_f1:.4f}")
    print(f"Test accuracy:      {test_metrics['accuracy']:.4f}")
    print(f"Test precision:     {test_metrics['precision']:.4f}")
    print(f"Test recall:        {test_metrics['recall']:.4f}")
    print(f"Test F1:            {test_metrics['f1']:.4f}")
    print(f"Test ROC-AUC:       {test_metrics['roc_auc']}")
    print(f"Saved model:        {best_model_path}")
    print(f"Grad-CAM outputs:   {gradcam_dir}")


if __name__ == "__main__":
    main()
