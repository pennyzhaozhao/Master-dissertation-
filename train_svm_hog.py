"""
Train and evaluate an RBF-kernel SVM classifier on cached HOG features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import threading
import time
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
from sklearn.svm import SVC
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
    parser = argparse.ArgumentParser(description="Train RBF SVM on HOG features from balanced splits.")
    parser.add_argument("--train_csv", required=True, type=Path)
    parser.add_argument("--val_csv", required=True, type=Path)
    parser.add_argument("--test_csv", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--image_size", default=224, type=int)
    parser.add_argument("--limit_train", default=None, type=int)
    parser.add_argument("--limit_val", default=None, type=int)
    parser.add_argument("--limit_test", default=None, type=int)
    parser.add_argument("--num_workers", default=max(1, (os.cpu_count() or 2) - 1), type=int)
    return parser.parse_args()


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_balanced_csv_name(path: Path, expected_name: str) -> None:
    if path.name.lower() != expected_name:
        raise ValueError(f"Expected {expected_name}, got {path}. Use the balanced CSVs only.")


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
    limited = pd.concat(parts, axis=0).sample(frac=1.0, random_state=SEED)
    return limited.head(limit)


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
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
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
            return (
                np.load(x_path),
                np.load(y_path),
                np.load(paths_path, allow_pickle=True).tolist(),
                [],
            )

    tasks = [
        (idx, str(Path(row.sample_path)), int(row.label), image_size)
        for idx, row in enumerate(df.itertuples(index=False))
    ]
    features: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[str] = []
    failures: list[str] = []

    if num_workers <= 1:
        iterator = map(extract_one_hog, tasks)
        results = tqdm(iterator, total=len(tasks), desc=f"Extracting HOG {split_name}")
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
        if num_workers > 1:
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

    return x, y, paths, failures


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


def save_plots(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray, plots_dir: Path, model_name: str) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Manipulated"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(plots_dir / f"{model_name}_confusion_matrix.png", dpi=200)
    plt.close()

    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_value = roc_auc_score(y_true, y_score)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC-AUC = {auc_value:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{model_name} ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(plots_dir / f"{model_name}_roc_curve.png", dpi=200)
        plt.close()
    except ValueError:
        print("ROC curve skipped because only one class is present in y_true.")


def fit_with_heartbeat(model: SVC, x: np.ndarray, y: np.ndarray, desc: str, tick_seconds: int = 5) -> float:
    """
    Fit SVC while showing a tqdm heartbeat.

    sklearn's libsvm backend does not expose fine-grained training progress, so
    this bar reports elapsed time while fit() is still running. It is meant to
    make long RBF-SVM fits visibly alive rather than silent.
    """
    stop_event = threading.Event()
    start_time = time.perf_counter()

    def heartbeat() -> None:
        with tqdm(total=None, desc=desc, unit="tick", dynamic_ncols=True) as pbar:
            while not stop_event.wait(tick_seconds):
                elapsed_minutes = (time.perf_counter() - start_time) / 60.0
                pbar.set_postfix_str(f"elapsed={elapsed_minutes:.1f} min")
                pbar.update(1)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        model.fit(x, y)
    finally:
        stop_event.set()
        thread.join()

    return time.perf_counter() - start_time


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

    x_train, y_train, _, train_failures = load_or_extract_features(
        "train", train_csv, train_df, args.image_size, cache_dir, args.num_workers
    )
    x_val, y_val, _, val_failures = load_or_extract_features(
        "val", val_csv, val_df, args.image_size, cache_dir, args.num_workers
    )
    x_test, y_test, _, test_failures = load_or_extract_features(
        "test", test_csv, test_df, args.image_size, cache_dir, args.num_workers
    )

    print(f"Feature shapes: train={x_train.shape}, val={x_val.shape}, test={x_test.shape}")

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)

    param_grid = [
        {"C": c_value, "gamma": gamma_value}
        for c_value in [0.1, 1, 10, 100]
        for gamma_value in ["scale", 0.001, 0.01, 0.1]
    ]

    best_params: dict[str, Any] | None = None
    best_f1 = -1.0
    validation_results = []

    for params in tqdm(param_grid, desc="SVM grid search", unit="combo", dynamic_ncols=True):
        c_value = params["C"]
        gamma_value = params["gamma"]
        tqdm.write(f"Training SVM combo: C={c_value}, gamma={gamma_value}")
        model = SVC(C=c_value, gamma=gamma_value, kernel="rbf", random_state=SEED)
        fit_seconds = fit_with_heartbeat(
            model,
            x_train_scaled,
            y_train,
            desc=f"fit C={c_value}, gamma={gamma_value}",
        )
        tqdm.write(f"Finished fit in {fit_seconds / 60.0:.1f} min. Predicting validation set...")
        val_pred = model.predict(x_val_scaled)
        val_f1 = f1_score(y_val, val_pred, zero_division=0)
        result = {
            "C": c_value,
            "gamma": gamma_value,
            "validation_f1": float(val_f1),
            "fit_seconds": float(fit_seconds),
        }
        validation_results.append(result)
        tqdm.write(f"C={c_value}, gamma={gamma_value}: validation F1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_params = {"C": c_value, "gamma": gamma_value}

    if best_params is None:
        raise RuntimeError("SVM grid search failed to select parameters.")

    print(
        f"Best SVM params: C={best_params['C']}, gamma={best_params['gamma']} "
        f"with validation F1={best_f1:.4f}"
    )

    x_train_val_scaled = np.vstack([x_train_scaled, x_val_scaled])
    y_train_val = np.concatenate([y_train, y_val])
    x_test_scaled = scaler.transform(x_test)

    final_model = SVC(
        C=best_params["C"],
        gamma=best_params["gamma"],
        kernel="rbf",
        random_state=SEED,
    )
    final_fit_seconds = fit_with_heartbeat(
        final_model,
        x_train_val_scaled,
        y_train_val,
        desc=f"final fit C={best_params['C']}, gamma={best_params['gamma']}",
    )
    print(f"Final fit completed in {final_fit_seconds / 60.0:.1f} min.")

    y_pred = final_model.predict(x_test_scaled)
    y_score = final_model.decision_function(x_test_scaled)
    test_metrics = compute_metrics(y_test, y_pred, y_score)

    joblib.dump(final_model, models_dir / "svm_hog_model.joblib")
    joblib.dump(scaler, models_dir / "svm_hog_scaler.joblib")
    save_plots(y_test, y_pred, y_score, plots_dir, "svm_hog")

    metrics_payload = {
        "model": "SVM_RBF_HOG",
        "seed": SEED,
        "image_size": args.image_size,
        "hog_params": {
            **HOG_PARAMS,
            "pixels_per_cell": list(HOG_PARAMS["pixels_per_cell"]),
            "cells_per_block": list(HOG_PARAMS["cells_per_block"]),
        },
        "best_params": best_params,
        "final_fit_seconds": float(final_fit_seconds),
        "validation_results": validation_results,
        "test_metrics": test_metrics,
        "feature_shapes": {
            "train": list(x_train.shape),
            "val": list(x_val.shape),
            "test": list(x_test.shape),
        },
        "failed_image_loads": {
            "train": len(train_failures),
            "val": len(val_failures),
            "test": len(test_failures),
        },
    }
    with (metrics_dir / "svm_hog_metrics.json").open("w", encoding="utf-8") as f:
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
