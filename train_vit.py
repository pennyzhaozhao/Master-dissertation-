"""
Fine-tune a pretrained Vision Transformer for binary facial manipulation detection.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

# Keep Transformers in PyTorch-only mode. This avoids importing a broken or
# unnecessary TensorFlow installation on Windows.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

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
from transformers import AutoImageProcessor, AutoModelForImageClassification


DEFAULT_CNN_FILTERED_DIR = Path("outputs/cnn/filtered_csvs")
DEFAULT_BALANCED_DIR = Path("outputs/dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained ViT for AI-generated facial manipulation detection."
    )
    parser.add_argument("--train_csv", default=None, type=Path)
    parser.add_argument("--val_csv", default=None, type=Path)
    parser.add_argument("--test_csv", default=None, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--image_size", default=224, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--limit_train", default=None, type=int)
    parser.add_argument("--limit_val", default=None, type=int)
    parser.add_argument("--limit_test", default=None, type=int)
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu.")
    parser.add_argument("--model_name", default="google/vit-base-patch16-224")
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


def default_csv_paths() -> tuple[Path, Path, Path, str]:
    cnn_paths = (
        DEFAULT_CNN_FILTERED_DIR / "train_filtered.csv",
        DEFAULT_CNN_FILTERED_DIR / "val_filtered.csv",
        DEFAULT_CNN_FILTERED_DIR / "test_filtered.csv",
    )
    if all(path.exists() for path in cnn_paths):
        return (*cnn_paths, "cnn_filtered")

    balanced_paths = (
        DEFAULT_BALANCED_DIR / "train_balanced.csv",
        DEFAULT_BALANCED_DIR / "val_balanced.csv",
        DEFAULT_BALANCED_DIR / "test_balanced.csv",
    )
    return (*balanced_paths, "balanced_fallback")


def resolve_csv_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, str]:
    default_train, default_val, default_test, source = default_csv_paths()
    train_csv = args.train_csv or default_train
    val_csv = args.val_csv or default_val
    test_csv = args.test_csv or default_test
    if args.train_csv or args.val_csv or args.test_csv:
        source = "user_provided"
    return train_csv.resolve(), val_csv.resolve(), test_csv.resolve(), source


def stratified_limit(df: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    if limit <= 0:
        raise ValueError("Limit values must be positive.")
    parts = []
    labels = sorted(df["label"].unique().tolist())
    remaining = limit
    for i, label in enumerate(labels):
        label_df = df[df["label"] == label]
        if i == len(labels) - 1:
            take = min(len(label_df), remaining)
        else:
            take = min(len(label_df), max(1, int(round(limit * len(label_df) / len(df)))))
        if take > 0:
            parts.append(label_df.sample(n=take, random_state=seed))
        remaining -= take
    if not parts:
        return df.head(0).copy()
    return pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed).head(limit).reset_index(drop=True)


def is_readable_image(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (OSError, UnidentifiedImageError, ValueError):
        return False


def read_filter_save_csv(
    csv_path: Path,
    split_name: str,
    output_dir: Path,
    limit: int | None,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"{split_name} CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = {"sample_path", "label"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing_columns)}")

    df["label"] = df["label"].astype(int)
    unexpected = set(df["label"].unique()).difference({0, 1})
    if unexpected:
        raise ValueError(f"{csv_path} contains labels outside {{0, 1}}: {sorted(unexpected)}")

    original_rows = len(df)
    if limit is not None and limit < len(df):
        df = stratified_limit(df, limit, seed)

    checked_rows = len(df)
    exists_mask = df["sample_path"].astype(str).map(lambda p: Path(p).exists())
    missing_df = df[~exists_mask].copy()
    missing_removed = int((~exists_mask).sum())
    df = df[exists_mask].copy()

    readable_mask = df["sample_path"].astype(str).map(is_readable_image)
    corrupted_df = df[~readable_mask].copy()
    corrupted_removed = int((~readable_mask).sum())
    df = df[readable_mask].copy().reset_index(drop=True)

    removed_records = []
    if len(missing_df):
        tmp = missing_df.copy()
        tmp["removal_reason"] = "missing_path"
        removed_records.append(tmp)
    if len(corrupted_df):
        tmp = corrupted_df.copy()
        tmp["removal_reason"] = "corrupted_or_unreadable_image"
        removed_records.append(tmp)
    if removed_records:
        removed_df = pd.concat(removed_records, axis=0)
    else:
        removed_df = pd.DataFrame(columns=list(df.columns) + ["removal_reason"])

    filtered_path = output_dir / "filtered_csvs" / f"{split_name}_filtered.csv"
    removed_path = output_dir / "filtered_csvs" / f"{split_name}_removed_files.csv"
    df.to_csv(filtered_path, index=False, encoding="utf-8-sig")
    removed_df.to_csv(removed_path, index=False, encoding="utf-8-sig")

    stats = {
        "original_rows": int(original_rows),
        "checked_rows": int(checked_rows),
        "missing_removed": int(missing_removed),
        "corrupted_removed": int(corrupted_removed),
        "after": int(len(df)),
        "filtered_csv": str(filtered_path.resolve()),
        "removed_files_csv": str(removed_path.resolve()),
    }
    return df, stats


def print_class_counts(split_name: str, df: pd.DataFrame, stats: dict[str, int]) -> None:
    counts = df["label"].value_counts().to_dict()
    print(
        f"{split_name}: Real={counts.get(0, 0)} | Manipulated={counts.get(1, 0)} | "
        f"Total={len(df)} | missing removed={stats['missing_removed']} | "
        f"corrupted removed={stats['corrupted_removed']}"
    )


class ViTDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: transforms.Compose) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        sample_path = str(row["sample_path"])
        with Image.open(sample_path) as img:
            image = ImageOps.exif_transpose(img).convert("RGB")
        image = self.transform(image)
        return {
            "pixel_values": image,
            "labels": torch.tensor(int(row["label"]), dtype=torch.long),
            "sample_path": sample_path,
        }


def make_transforms(image_processor: Any, image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    mean = getattr(image_processor, "image_mean", [0.485, 0.456, 0.406])
    std = getattr(image_processor, "image_std", [0.229, 0.224, 0.225])
    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_tfms, eval_tfms


def worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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
        ViTDataset(df, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
        generator=generator,
    )


def load_vit_model(model_name: str) -> tuple[Any, AutoModelForImageClassification]:
    try:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=2,
            id2label={0: "Real", 1: "Manipulated"},
            label2id={"Real": 0, "Manipulated": 1},
            ignore_mismatched_sizes=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not load the pretrained ViT model/image processor. "
            "If this is the first run, Hugging Face may need internet access to download "
            f"{model_name}; otherwise make sure it is already cached locally. "
            f"Original error: {exc}"
        ) from exc
    return processor, model


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
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
    epoch: int,
    use_amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    progress = tqdm(loader, desc=f"Train epoch {epoch}", unit="batch", dynamic_ncols=True)
    for batch in progress:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(pixel_values=pixel_values).logits
            loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
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
    use_amp: bool,
) -> tuple[float, np.ndarray, np.ndarray, list[str]]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_paths: list[str] = []

    for batch in tqdm(loader, desc=desc, unit="batch", dynamic_ncols=True):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(pixel_values=pixel_values).logits
            loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)[:, 1]

        batch_size = labels.size(0)
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
    plt.title("ViT Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "vit_loss_curve.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(history_df["epoch"], history_df["val_f1"], label="Val F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title("ViT Validation F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "vit_f1_curve.png", dpi=200)
    plt.close()


def save_test_plots(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, plots_dir: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Manipulated"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("ViT Confusion Matrix")
    plt.tight_layout()
    plt.savefig(plots_dir / "vit_confusion_matrix.png", dpi=200)
    plt.close()

    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_value = roc_auc_score(y_true, y_prob)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC-AUC = {auc_value:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ViT ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(plots_dir / "vit_roc_curve.png", dpi=200)
        plt.close()
    except ValueError:
        print("ROC curve skipped because only one class is present.")


def save_json(data: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    use_amp = device.type == "cuda"

    output_dir = args.output_dir.expanduser().resolve()
    models_dir = output_dir / "models"
    metrics_dir = output_dir / "metrics"
    plots_dir = output_dir / "plots"
    predictions_dir = output_dir / "predictions"
    filtered_dir = output_dir / "filtered_csvs"
    for folder in [models_dir, metrics_dir, plots_dir, predictions_dir, filtered_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    train_csv, val_csv, test_csv, csv_source = resolve_csv_paths(args)
    print(f"CSV source: {csv_source}")
    print(f"Train CSV: {train_csv}")
    print(f"Val CSV:   {val_csv}")
    print(f"Test CSV:  {test_csv}")

    train_df, train_stats = read_filter_save_csv(train_csv, "train", output_dir, args.limit_train, args.seed)
    val_df, val_stats = read_filter_save_csv(val_csv, "val", output_dir, args.limit_val, args.seed)
    test_df, test_stats = read_filter_save_csv(test_csv, "test", output_dir, args.limit_test, args.seed)

    print("Class counts after filtering")
    print_class_counts("train", train_df, train_stats)
    print_class_counts("val", val_df, val_stats)
    print_class_counts("test", test_df, test_stats)

    image_processor, model = load_vit_model(args.model_name)
    model.to(device)

    train_tfms, eval_tfms = make_transforms(image_processor, args.image_size)
    train_loader = make_loader(train_df, train_tfms, args.batch_size, True, args.num_workers, args.seed)
    val_loader = make_loader(val_df, eval_tfms, args.batch_size, False, args.num_workers, args.seed)
    test_loader = make_loader(test_df, eval_tfms, args.batch_size, False, args.num_workers, args.seed)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    best_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    patience = 3
    best_model_dir = models_dir / "vit_best_model"
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, use_amp)
        val_loss, val_true, val_prob, _ = evaluate(
            model, val_loader, criterion, device, f"Validate epoch {epoch}", use_amp
        )
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
            model.save_pretrained(best_model_dir)
            image_processor.save_pretrained(best_model_dir)
            torch.save(
                {
                    "epoch": epoch,
                    "best_f1": best_f1,
                    "model_name": args.model_name,
                    "image_size": args.image_size,
                    "seed": args.seed,
                },
                best_model_dir / "training_state.pt",
            )
            print(f"Saved best model: {best_model_dir}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    history_df = pd.DataFrame(history)
    history_csv = metrics_dir / "vit_training_history.csv"
    history_json = metrics_dir / "vit_training_history.json"
    history_df.to_csv(history_csv, index=False, encoding="utf-8-sig")
    save_json(history, history_json)
    save_curves(history_df, plots_dir)

    model = AutoModelForImageClassification.from_pretrained(best_model_dir).to(device)
    test_loss, y_true, y_prob, sample_paths = evaluate(model, test_loader, criterion, device, "Testing", use_amp)
    y_pred = (y_prob >= 0.5).astype(int)
    test_metrics = compute_binary_metrics(y_true, y_prob)
    test_metrics["test_loss"] = float(test_loss)
    test_metrics["best_epoch"] = int(best_epoch)
    test_metrics["best_validation_f1"] = float(best_f1)

    predictions_df = pd.DataFrame(
        {
            "sample_path": sample_paths,
            "true_label": y_true.astype(int),
            "probability_manipulated": y_prob.astype(float),
            "predicted_label": y_pred.astype(int),
        }
    )
    predictions_csv = predictions_dir / "vit_test_predictions.csv"
    metrics_json = metrics_dir / "vit_test_metrics.json"
    metrics_csv = metrics_dir / "vit_test_metrics.csv"
    predictions_df.to_csv(predictions_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame([test_metrics]).to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    save_json(test_metrics, metrics_json)
    save_test_plots(y_true, y_pred, y_prob, plots_dir)

    summary = {
        "model_name": args.model_name,
        "csv_source": csv_source,
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "test_csv": str(test_csv),
        "filtering": {
            "train": train_stats,
            "val": val_stats,
            "test": test_stats,
        },
        "best_validation_f1": float(best_f1),
        "best_epoch": int(best_epoch),
        "test_metrics": test_metrics,
        "saved_model_path": str(best_model_dir.resolve()),
        "saved_metrics_path": str(metrics_json.resolve()),
        "saved_predictions_path": str(predictions_csv.resolve()),
    }
    save_json(summary, metrics_dir / "vit_run_summary.json")

    print("\nViT fine-tuning complete")
    print(f"Best validation F1: {best_f1:.4f}")
    print(f"Test accuracy:      {test_metrics['accuracy']:.4f}")
    print(f"Test precision:     {test_metrics['precision']:.4f}")
    print(f"Test recall:        {test_metrics['recall']:.4f}")
    print(f"Test F1:            {test_metrics['f1']:.4f}")
    print(f"Test ROC-AUC:       {test_metrics['roc_auc']}")
    print(f"Saved model path:   {best_model_dir}")
    print(f"Saved metrics path: {metrics_json}")


if __name__ == "__main__":
    main()
