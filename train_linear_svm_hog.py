"""
Large-scale Linear SVM baseline on cached HOG features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError

SKIMAGE_CACHE_DIR = Path(__file__).resolve().parent / ".skimage_cache"
SKIMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("SKIMAGE_DATADIR", str(SKIMAGE_CACHE_DIR))

from skimage.feature import hog
from sklearn.linear_model import SGDClassifier
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
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


SEED = 42
HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (16, 16),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
    "transform_sqrt": True,
    "feature_vector": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train large-scale Linear SVM on HOG features.")
    parser.add_argument("--train_csv", required=True, type=Path)
    parser.add_argument("--val_csv", required=True, type=Path)
    parser.add_argument("--test_csv", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--image_size", default=224, type=int)
    parser.add_argument("--limit_train", default=None, type=int)
    parser.add_argument("--limit_val", default=None, type=int)
    parser.add_argument("--limit_test", default=None, type=int)
    parser.add_argument("--num_workers", default=max(1, (os.cpu_count() or 2) - 1), type=int)
    parser.add_argument("--batch_size", default=4096, type=int)
    parser.add_argument("--epochs", default=8, type=int)
    parser.add_argument(
        "--alpha_values",
        default="0.00001,0.0001,0.001,0.01",
        help="Comma-separated SGD regularisation strengths to tune on validation F1.",
    )
    return parser.parse_args()


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_balanced_csv_name(path: Path, expected_name: str) -> None:
    if path.name.lower() != expected_name:
        raise ValueError(f"Expected {expected_name}, got {path}. Use the balanced CSVs only.")


def stratified_limit(df: pd.DataFrame, limit: int) -> pd.DataFrame:
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
        parts.append(label_df.sample(n=take, random_state=SEED))
        remaining -= take
    return pd.concat(parts, axis=0).sample(frac=1.0, random_state=SEED).head(limit)


def read_split_csv(path: Path, expected_name: str, limit: int | None) -> pd.DataFrame:
    ensure_balanced_csv_name(path, expected_name)
    df = pd.read_csv(path)
    required = {"sample_path", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    df["label"] = df["label"].astype(int)
    unexpected = set(df["label"].unique()).difference({0, 1})
    if unexpected:
        raise ValueError(f"{path} contains labels outside {{0, 1}}: {sorted(unexpected)}")
    if limit is not None and limit < len(df):
        df = stratified_limit(df, limit)
    return df.reset_index(drop=True)


def print_class_counts(name: str, df: pd.DataFrame) -> None:
    counts = df["label"].value_counts().to_dict()
    print(f"{name}: Real={counts.get(0, 0)} | Manipulated={counts.get(1, 0)} | Total={len(df)}")


def cache_key(csv_path: Path, df: pd.DataFrame, image_size: int) -> tuple[str, dict[str, Any]]:
    stat = csv_path.stat()
    sample_fingerprint = hashlib.sha256(
        "\n".join(df["sample_path"].astype(str).tolist()).encode("utf-8")
    ).hexdigest()
    meta = {
        "csv_path": str(csv_path.resolve()),
        "csv_size": stat.st_size,
        "csv_mtime_ns": stat.st_mtime_ns,
        "n_rows": int(len(df)),
        "sample_path_sha256": sample_fingerprint,
        "image_size": image_size,
        "hog_params": {
            **HOG_PARAMS,
            "pixels_per_cell": list(HOG_PARAMS["pixels_per_cell"]),
            "cells_per_block": list(HOG_PARAMS["cells_per_block"]),
        },
    }
    raw = json.dumps(meta, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16], meta


def extract_one_hog(task: tuple[int, str, int, int]) -> tuple[int, np.ndarray | None, str | None]:
    idx, image_path, label, image_size = task
    try:
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert("L")
            if img.size != (image_size, image_size):
                img = img.resize((image_size, image_size), Image.Resampling.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
        feature = hog(arr, **HOG_PARAMS).astype(np.float32)
        return idx, feature, None
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        return idx, None, f"{image_path}: {exc}"


def load_or_extract_features(
    split_name: str,
    csv_path: Path,
    df: pd.DataFrame,
    image_size: int,
    cache_dir: Path,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key, meta = cache_key(csv_path, df, image_size)
    prefix = cache_dir / f"{split_name}_{key}_hog_{image_size}"
    x_path = prefix.with_suffix(".X.npy")
    y_path = prefix.with_suffix(".y.npy")
    paths_path = prefix.with_suffix(".paths.npy")
    meta_path = prefix.with_suffix(".meta.json")

    if x_path.exists() and y_path.exists() and paths_path.exists() and meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            cached_meta = json.load(f)
        if cached_meta == meta:
            print(f"Loading cached HOG features for {split_name}: {x_path}")
            return np.load(x_path, mmap_mode="r"), np.load(y_path), np.load(paths_path, allow_pickle=True).tolist()

    tasks = [
        (idx, str(Path(row.sample_path)), int(row.label), image_size)
        for idx, row in enumerate(df.itertuples(index=False))
    ]
    features: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[str] = []
    failures: list[str] = []

    if num_workers <= 1:
        results = tqdm(map(extract_one_hog, tasks), total=len(tasks), desc=f"Extracting HOG {split_name}")
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=num_workers)
        results = tqdm(
            executor.map(extract_one_hog, tasks, chunksize=64),
            total=len(tasks),
            desc=f"Extracting HOG {split_name}",
        )

    try:
        for idx, feature, error in results:
            if feature is None:
                failures.append(error or f"Failed sample index {idx}")
                continue
            features.append(feature)
            labels.append(int(df.iloc[idx]["label"]))
            paths.append(str(df.iloc[idx]["sample_path"]))
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    if not features:
        raise RuntimeError(f"No HOG features could be extracted for {split_name}.")

    x = np.vstack(features).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    np.save(x_path, x)
    np.save(y_path, y)
    np.save(paths_path, np.asarray(paths, dtype=object), allow_pickle=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    if failures:
        failure_path = prefix.with_suffix(".failures.txt")
        failure_path.write_text("\n".join(failures), encoding="utf-8")
        print(f"{split_name}: {len(failures)} failed image loads. See {failure_path}")

    return np.load(x_path, mmap_mode="r"), y, paths


def batch_indices(n_rows: int, batch_size: int, shuffle: bool, seed: int) -> list[np.ndarray]:
    indices = np.arange(n_rows)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    return [indices[start : start + batch_size] for start in range(0, n_rows, batch_size)]


def fit_scaler_on_train(x_train: np.ndarray, batch_size: int) -> StandardScaler:
    scaler = StandardScaler()
    batches = batch_indices(len(x_train), batch_size, shuffle=False, seed=SEED)
    for idx in tqdm(batches, desc="Fitting StandardScaler", unit="batch", dynamic_ncols=True):
        scaler.partial_fit(np.asarray(x_train[idx], dtype=np.float32))
    return scaler


def train_sgd_linear_svm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    scaler: StandardScaler,
    alpha: float,
    epochs: int,
    batch_size: int,
    desc: str,
) -> SGDClassifier:
    model = SGDClassifier(
        loss="hinge",
        alpha=alpha,
        penalty="l2",
        learning_rate="optimal",
        max_iter=1,
        tol=None,
        shuffle=False,
        random_state=SEED,
        average=True,
    )

    classes = np.array([0, 1], dtype=np.int64)
    first_batch = True
    for epoch in tqdm(range(1, epochs + 1), desc=desc, unit="epoch", dynamic_ncols=True):
        batches = batch_indices(len(x_train), batch_size, shuffle=True, seed=SEED + epoch)
        inner = tqdm(batches, desc=f"{desc} epoch {epoch}/{epochs}", unit="batch", leave=False, dynamic_ncols=True)
        for idx in inner:
            x_batch = scaler.transform(np.asarray(x_train[idx], dtype=np.float32))
            y_batch = y_train[idx]
            if first_batch:
                model.partial_fit(x_batch, y_batch, classes=classes)
                first_batch = False
            else:
                model.partial_fit(x_batch, y_batch)
    return model


def train_sgd_linear_svm_from_splits(
    feature_splits: list[np.ndarray],
    label_splits: list[np.ndarray],
    scaler: StandardScaler,
    alpha: float,
    epochs: int,
    batch_size: int,
    desc: str,
) -> SGDClassifier:
    """Train across multiple feature arrays without concatenating them in memory."""
    model = SGDClassifier(
        loss="hinge",
        alpha=alpha,
        penalty="l2",
        learning_rate="optimal",
        max_iter=1,
        tol=None,
        shuffle=False,
        random_state=SEED,
        average=True,
    )

    classes = np.array([0, 1], dtype=np.int64)
    first_batch = True
    for epoch in tqdm(range(1, epochs + 1), desc=desc, unit="epoch", dynamic_ncols=True):
        split_order = list(range(len(feature_splits)))
        random.Random(SEED + epoch).shuffle(split_order)
        total_batches = sum(
            (len(feature_splits[split_idx]) + batch_size - 1) // batch_size
            for split_idx in split_order
        )
        inner = tqdm(total=total_batches, desc=f"{desc} epoch {epoch}/{epochs}", unit="batch", leave=False, dynamic_ncols=True)
        for split_idx in split_order:
            x_split = feature_splits[split_idx]
            y_split = label_splits[split_idx]
            for idx in batch_indices(len(x_split), batch_size, shuffle=True, seed=SEED + epoch + split_idx):
                x_batch = scaler.transform(np.asarray(x_split[idx], dtype=np.float32))
                y_batch = y_split[idx]
                if first_batch:
                    model.partial_fit(x_batch, y_batch, classes=classes)
                    first_batch = False
                else:
                    model.partial_fit(x_batch, y_batch)
                inner.update(1)
        inner.close()
    return model


def predict_in_batches(
    model: SGDClassifier,
    scaler: StandardScaler,
    x: np.ndarray,
    batch_size: int,
    desc: str,
) -> tuple[np.ndarray, np.ndarray]:
    preds: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    batches = batch_indices(len(x), batch_size, shuffle=False, seed=SEED)
    for idx in tqdm(batches, desc=desc, unit="batch", dynamic_ncols=True):
        x_batch = scaler.transform(np.asarray(x[idx], dtype=np.float32))
        preds.append(model.predict(x_batch))
        scores.append(model.decision_function(x_batch))
    return np.concatenate(preds), np.concatenate(scores)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
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
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


def save_plots(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Manipulated"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("linear_svm_hog Confusion Matrix")
    plt.tight_layout()
    plt.savefig(plots_dir / "linear_svm_hog_confusion_matrix.png", dpi=200)
    plt.close()

    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_value = roc_auc_score(y_true, y_score)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC-AUC = {auc_value:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("linear_svm_hog ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(plots_dir / "linear_svm_hog_roc_curve.png", dpi=200)
        plt.close()
    except ValueError:
        print("ROC curve skipped because only one class is present in y_true.")


def parse_alpha_values(value: str) -> list[float]:
    alphas = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not alphas:
        raise ValueError("--alpha_values must contain at least one value.")
    return alphas


def main() -> None:
    args = parse_args()
    set_seed(SEED)

    output_dir = args.output_dir.expanduser().resolve()
    models_dir = output_dir / "models"
    metrics_dir = output_dir / "metrics"
    plots_dir = output_dir / "plots"
    cache_dir = output_dir / "feature_cache" / "hog"
    for folder in [models_dir, metrics_dir, plots_dir, cache_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    train_csv = args.train_csv.expanduser().resolve()
    val_csv = args.val_csv.expanduser().resolve()
    test_csv = args.test_csv.expanduser().resolve()

    train_df = read_split_csv(train_csv, "train_balanced.csv", args.limit_train)
    val_df = read_split_csv(val_csv, "val_balanced.csv", args.limit_val)
    test_df = read_split_csv(test_csv, "test_balanced.csv", args.limit_test)

    print("Class counts")
    print_class_counts("train", train_df)
    print_class_counts("val", val_df)
    print_class_counts("test", test_df)

    x_train, y_train, _ = load_or_extract_features("train", train_csv, train_df, args.image_size, cache_dir, args.num_workers)
    x_val, y_val, _ = load_or_extract_features("val", val_csv, val_df, args.image_size, cache_dir, args.num_workers)
    x_test, y_test, _ = load_or_extract_features("test", test_csv, test_df, args.image_size, cache_dir, args.num_workers)

    print(f"Feature shapes: train={x_train.shape}, val={x_val.shape}, test={x_test.shape}")

    scaler = fit_scaler_on_train(x_train, args.batch_size)

    alpha_values = parse_alpha_values(args.alpha_values)
    validation_results = []
    best_alpha = None
    best_f1 = -1.0

    for alpha in tqdm(alpha_values, desc="Linear SVM alpha search", unit="alpha", dynamic_ncols=True):
        tqdm.write(f"Training Linear SVM alpha={alpha}")
        model = train_sgd_linear_svm(
            x_train=x_train,
            y_train=y_train,
            scaler=scaler,
            alpha=alpha,
            epochs=args.epochs,
            batch_size=args.batch_size,
            desc=f"alpha={alpha}",
        )
        val_pred, val_score = predict_in_batches(model, scaler, x_val, args.batch_size, f"Validating alpha={alpha}")
        val_f1 = f1_score(y_val, val_pred, zero_division=0)
        val_auc = float(roc_auc_score(y_val, val_score)) if len(np.unique(y_val)) == 2 else None
        validation_results.append({"alpha": alpha, "validation_f1": float(val_f1), "validation_roc_auc": val_auc})
        tqdm.write(f"alpha={alpha}: validation F1={val_f1:.4f}, ROC-AUC={val_auc}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_alpha = alpha

    if best_alpha is None:
        raise RuntimeError("No alpha value was selected.")

    print(f"Best alpha: {best_alpha} with validation F1={best_f1:.4f}")

    final_model = train_sgd_linear_svm_from_splits(
        feature_splits=[x_train, x_val],
        label_splits=[y_train, y_val],
        scaler=scaler,
        alpha=float(best_alpha),
        epochs=args.epochs,
        batch_size=args.batch_size,
        desc="final train+val",
    )

    y_pred, y_score = predict_in_batches(final_model, scaler, x_test, args.batch_size, "Testing")
    test_metrics = compute_metrics(y_test, y_pred, y_score)

    joblib.dump(final_model, models_dir / "linear_svm_hog_model.joblib")
    joblib.dump(scaler, models_dir / "linear_svm_hog_scaler.joblib")
    save_plots(y_test, y_pred, y_score, plots_dir)

    metrics_payload = {
        "model": "Linear_SVM_HOG_SGDClassifier",
        "seed": SEED,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "alpha_values": alpha_values,
        "best_alpha": best_alpha,
        "validation_results": validation_results,
        "test_metrics": test_metrics,
        "feature_shapes": {
            "train": list(x_train.shape),
            "val": list(x_val.shape),
            "test": list(x_test.shape),
        },
        "note": "StandardScaler was fitted on training features only. Final classifier was trained on train+val using that scaler.",
    }
    with (metrics_dir / "linear_svm_hog_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    print("\nTest metrics")
    print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1:        {test_metrics['f1']:.4f}")
    print(f"ROC-AUC:   {test_metrics['roc_auc']}")
    print(f"Saved model, metrics, and plots under: {output_dir}")


if __name__ == "__main__":
    main()
