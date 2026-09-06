"""
Compare spatial explanations for KNN-HOG, Linear SVM-HOG, CNN, and ViT models.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
SKIMAGE_CACHE_DIR = Path(__file__).resolve().parent / ".skimage_cache"
SKIMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("SKIMAGE_DATADIR", str(SKIMAGE_CACHE_DIR))

import cv2
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from skimage.feature import hog
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModelForImageClassification

from train_cnn_gradcam import GradCAM, IMAGENET_MEAN, IMAGENET_STD, SimpleCNN
from train_knn_hog import HOG_PARAMS


LABEL_NAMES = {0: "Real", 1: "Manipulated"}
MODEL_ORDER = ["knn", "linear_svm", "cnn", "vit"]


@dataclass
class Config:
    root_dir: Path
    output_dir: Path
    test_csv: Path
    knn_model_path: Path
    knn_scaler_path: Path
    linear_svm_model_path: Path
    linear_svm_scaler_path: Path
    cnn_checkpoint_path: Path
    cnn_predictions_csv: Path
    vit_model_dir: Path
    vit_predictions_csv: Path
    image_size: int
    patch_size: int
    stride: int
    max_per_category: int
    seed: int
    device: str
    smoke: bool
    qualitative_examples: bool
    qualitative_per_label: int
    deletion_fractions: tuple[float, ...]


def parse_args() -> Config:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate model explainability comparison figures.")
    parser.add_argument("--test_csv", default=root / "outputs" / "cnn" / "filtered_csvs" / "test_filtered.csv", type=Path)
    parser.add_argument("--output_dir", default=root / "outputs" / "explainability_comparison", type=Path)
    parser.add_argument("--knn_model_path", default=root / "outputs" / "models" / "knn_hog_model.joblib", type=Path)
    parser.add_argument("--knn_scaler_path", default=root / "outputs" / "models" / "knn_hog_scaler.joblib", type=Path)
    parser.add_argument("--linear_svm_model_path", default=root / "outputs" / "models" / "linear_svm_hog_model.joblib", type=Path)
    parser.add_argument("--linear_svm_scaler_path", default=root / "outputs" / "models" / "linear_svm_hog_scaler.joblib", type=Path)
    parser.add_argument("--cnn_checkpoint_path", default=root / "outputs" / "cnn" / "models" / "cnn_best_model.pt", type=Path)
    parser.add_argument("--cnn_predictions_csv", default=root / "outputs" / "cnn" / "metrics" / "cnn_test_predictions.csv", type=Path)
    parser.add_argument("--vit_model_dir", default=root / "outputs" / "vit" / "models" / "vit_best_model", type=Path)
    parser.add_argument("--vit_predictions_csv", default=root / "outputs" / "vit" / "predictions" / "vit_test_predictions.csv", type=Path)
    parser.add_argument("--image_size", default=224, type=int)
    parser.add_argument("--patch_size", default=32, type=int)
    parser.add_argument("--stride", default=32, type=int)
    parser.add_argument("--max_per_category", default=5, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--smoke", action="store_true", help="Run only one TP and one TN image.")
    parser.add_argument(
        "--qualitative_examples",
        action="store_true",
        help="Generate representative Real/Manipulated cross-model heatmap figures only.",
    )
    parser.add_argument("--qualitative_per_label", default=3, type=int, help="Samples per true-label group.")
    parser.add_argument("--deletion_fractions", default="0.10,0.20,0.30")
    args = parser.parse_args()
    fractions = tuple(float(x.strip()) for x in args.deletion_fractions.split(",") if x.strip())
    return Config(
        root_dir=root,
        output_dir=args.output_dir.resolve(),
        test_csv=args.test_csv.resolve(),
        knn_model_path=args.knn_model_path.resolve(),
        knn_scaler_path=args.knn_scaler_path.resolve(),
        linear_svm_model_path=args.linear_svm_model_path.resolve(),
        linear_svm_scaler_path=args.linear_svm_scaler_path.resolve(),
        cnn_checkpoint_path=args.cnn_checkpoint_path.resolve(),
        cnn_predictions_csv=args.cnn_predictions_csv.resolve(),
        vit_model_dir=args.vit_model_dir.resolve(),
        vit_predictions_csv=args.vit_predictions_csv.resolve(),
        image_size=args.image_size,
        patch_size=args.patch_size,
        stride=args.stride,
        max_per_category=args.max_per_category,
        seed=args.seed,
        device=args.device,
        smoke=args.smoke,
        qualitative_examples=args.qualitative_examples,
        qualitative_per_label=args.qualitative_per_label,
        deletion_fractions=fractions,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def ensure_outputs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "figures": output_dir / "figures",
        "raw": output_dir / "raw_importance",
        "aggregate": output_dir / "aggregate",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def load_rgb(path: str | Path, image_size: int) -> np.ndarray:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img = img.resize((image_size, image_size), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def confusion_category(true_label: int, pred_label: int) -> str:
    if true_label == 1 and pred_label == 1:
        return "TP"
    if true_label == 0 and pred_label == 0:
        return "TN"
    if true_label == 0 and pred_label == 1:
        return "FP"
    return "FN"


def safe_stem(path: str) -> str:
    p = Path(path)
    return f"{p.parent.name}_{p.stem}".replace(" ", "_").replace(":", "")


def read_predictions(path: Path, score_column: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["sample_path"] = df["sample_path"].astype(str)
    df["true_label"] = df["true_label"].astype(int)
    df["predicted_label"] = df["predicted_label"].astype(int)
    df["score"] = df[score_column].astype(float)
    df["category"] = [confusion_category(t, p) for t, p in zip(df.true_label, df.predicted_label)]
    return df[["sample_path", "true_label", "predicted_label", "score", "category"]]


def select_shared_images(cnn_pred: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rng_seed = cfg.seed
    selected = []
    n = 1 if cfg.smoke else cfg.max_per_category
    categories = ["TP", "TN"] if cfg.smoke else ["TP", "TN", "FP", "FN"]
    for cat in categories:
        cat_df = cnn_pred[cnn_pred["category"] == cat]
        if len(cat_df) == 0:
            continue
        take = min(n, len(cat_df))
        selected.append(cat_df.sample(n=take, random_state=rng_seed))
    if not selected:
        raise RuntimeError("No shared images could be selected from CNN predictions.")
    result = pd.concat(selected).sample(frac=1.0, random_state=rng_seed).reset_index(drop=True)
    result["selection_category"] = result["category"]
    return result


def select_qualitative_examples(test_df: pd.DataFrame, cfg: Config) -> dict[int, pd.DataFrame]:
    selections: dict[int, pd.DataFrame] = {}
    n = max(1, cfg.qualitative_per_label)
    for label in [0, 1]:
        label_df = test_df[test_df["label"].astype(int) == label].copy()
        if len(label_df) == 0:
            raise RuntimeError(f"No test samples found for label {label} ({LABEL_NAMES[label]}).")
        take = min(n, len(label_df))
        selections[label] = label_df.sample(n=take, random_state=cfg.seed + label).reset_index(drop=True)
    return selections


def hog_feature_from_rgb(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return hog(gray, **HOG_PARAMS).astype(np.float32)


def make_hog_scorer(model: Any, scaler: Any) -> Callable[[np.ndarray], float]:
    proba_index = None
    if hasattr(model, "classes_") and hasattr(model, "predict_proba"):
        matches = np.where(np.asarray(model.classes_) == 1)[0]
        if len(matches):
            proba_index = int(matches[0])

    def score(rgb: np.ndarray) -> float:
        x = hog_feature_from_rgb(rgb).reshape(1, -1)
        x_scaled = scaler.transform(x)
        if proba_index is not None:
            return float(model.predict_proba(x_scaled)[0, proba_index])
        if hasattr(model, "decision_function"):
            value = np.asarray(model.decision_function(x_scaled)).reshape(-1)[0]
            if hasattr(model, "classes_") and list(model.classes_)[-1] != 1:
                value = -value
            return float(value)
        pred = int(model.predict(x_scaled)[0])
        return float(pred)

    return score


def make_hog_predictor(model: Any, scaler: Any) -> Callable[[np.ndarray], tuple[int, float]]:
    score_fn = make_hog_scorer(model, scaler)

    def predict(rgb: np.ndarray) -> tuple[int, float]:
        x = hog_feature_from_rgb(rgb).reshape(1, -1)
        x_scaled = scaler.transform(x)
        pred = int(model.predict(x_scaled)[0])
        return pred, score_fn(rgb)

    return predict


def cnn_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def make_cnn_scorer(model: SimpleCNN, device: torch.device, image_size: int) -> Callable[[np.ndarray], float]:
    tfm = cnn_eval_transform(image_size)

    @torch.no_grad()
    def score(rgb: np.ndarray) -> float:
        tensor = tfm(rgb).unsqueeze(0).to(device)
        return float(torch.sigmoid(model(tensor))[0].detach().cpu().item())

    return score


def make_cnn_predictor(model: SimpleCNN, device: torch.device, image_size: int) -> Callable[[np.ndarray], tuple[int, float]]:
    score_fn = make_cnn_scorer(model, device, image_size)

    def predict(rgb: np.ndarray) -> tuple[int, float]:
        score = score_fn(rgb)
        return int(score >= 0.5), score

    return predict


def vit_transform(mean: list[float], std: list[float], image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def read_vit_mean_std(model_dir: Path) -> tuple[list[float], list[float]]:
    config_path = model_dir / "preprocessor_config.json"
    if not config_path.exists():
        return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data.get("image_mean", [0.5, 0.5, 0.5])), list(data.get("image_std", [0.5, 0.5, 0.5]))


def make_vit_scorer(
    model: Any,
    device: torch.device,
    image_size: int,
    mean: list[float],
    std: list[float],
) -> Callable[[np.ndarray], float]:
    tfm = vit_transform(mean, std, image_size)

    @torch.no_grad()
    def score(rgb: np.ndarray) -> float:
        tensor = tfm(rgb).unsqueeze(0).to(device)
        logits = model(pixel_values=tensor).logits
        return float(torch.softmax(logits, dim=1)[0, 1].detach().cpu().item())

    return score


def make_vit_predictor(
    model: Any,
    device: torch.device,
    image_size: int,
    mean: list[float],
    std: list[float],
) -> Callable[[np.ndarray], tuple[int, float]]:
    score_fn = make_vit_scorer(model, device, image_size, mean, std)

    def predict(rgb: np.ndarray) -> tuple[int, float]:
        score = score_fn(rgb)
        return int(score >= 0.5), score

    return predict


def occlusion_heatmap(
    rgb: np.ndarray,
    score_fn: Callable[[np.ndarray], float],
    patch_size: int,
    stride: int,
    fill_rgb: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, float]:
    h, w = rgb.shape[:2]
    ys = list(range(0, max(h - patch_size, 0) + 1, stride))
    xs = list(range(0, max(w - patch_size, 0) + 1, stride))
    if ys[-1] != h - patch_size:
        ys.append(h - patch_size)
    if xs[-1] != w - patch_size:
        xs.append(w - patch_size)

    original_score = score_fn(rgb)
    grid = np.zeros((len(ys), len(xs)), dtype=np.float32)
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            occluded = rgb.copy()
            occluded[y : y + patch_size, x : x + patch_size, :] = fill_rgb
            grid[yi, xi] = original_score - score_fn(occluded)
    resized = cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
    return grid, resized.astype(np.float32), float(original_score)


def score_for_target_class(score_fn: Callable[[np.ndarray], float], target_label: int) -> Callable[[np.ndarray], float]:
    """Orient a binary Manipulated-class score toward the requested class."""

    if target_label == 1:
        return score_fn

    def real_score(rgb: np.ndarray) -> float:
        return -float(score_fn(rgb))

    return real_score


def mask_top_fraction(
    rgb: np.ndarray,
    importance: np.ndarray,
    fraction: float,
    fill_rgb: tuple[int, int, int],
) -> np.ndarray:
    masked = rgb.copy()
    flat = importance.reshape(-1)
    k = max(1, int(round(len(flat) * fraction)))
    threshold = np.partition(flat, len(flat) - k)[len(flat) - k]
    mask = importance >= threshold
    masked[mask] = fill_rgb
    return masked


def deletion_curve(
    rgb: np.ndarray,
    importance: np.ndarray,
    score_fn: Callable[[np.ndarray], float],
    fill_rgb: tuple[int, int, int],
    steps: int = 20,
) -> tuple[list[float], list[float], float]:
    fractions = np.linspace(0.0, 1.0, steps + 1)
    scores = []
    for frac in fractions:
        if frac == 0:
            masked = rgb
        else:
            masked = mask_top_fraction(rgb, importance, float(frac), fill_rgb)
        scores.append(float(score_fn(masked)))
    auc = float(np.trapz(scores, fractions))
    return fractions.tolist(), scores, auc


def cnn_gradcam(model: SimpleCNN, rgb: np.ndarray, device: torch.device, image_size: int) -> np.ndarray:
    tfm = cnn_eval_transform(image_size)
    tensor = tfm(rgb)
    gradcam = GradCAM(model, model.final_conv_layer)
    try:
        return gradcam.generate(tensor, device).astype(np.float32)
    finally:
        gradcam.remove_hooks()


def vit_attention_rollout(
    model: Any,
    rgb: np.ndarray,
    device: torch.device,
    image_size: int,
    mean: list[float],
    std: list[float],
) -> np.ndarray:
    tfm = vit_transform(mean, std, image_size)
    tensor = tfm(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(pixel_values=tensor, output_attentions=True)
    attentions = outputs.attentions
    if attentions is None:
        raise RuntimeError("ViT did not return attentions. Use an eager attention implementation if needed.")
    rollout = torch.eye(attentions[0].shape[-1], device=device)
    for attention in attentions:
        attn = attention[0].mean(dim=0)
        attn = attn + torch.eye(attn.shape[0], device=device)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        rollout = attn @ rollout
    cls_attention = rollout[0, 1:].detach().cpu().numpy()
    side = int(np.sqrt(cls_attention.shape[0]))
    heatmap = cls_attention[: side * side].reshape(side, side)
    heatmap = heatmap - heatmap.min()
    denom = heatmap.max()
    if denom > 0:
        heatmap = heatmap / denom
    return cv2.resize(heatmap, (image_size, image_size), interpolation=cv2.INTER_CUBIC).astype(np.float32)


def normalise_for_display(maps: list[np.ndarray]) -> tuple[float, float]:
    stack = np.stack(maps)
    vmin = float(np.nanpercentile(stack, 1))
    vmax = float(np.nanpercentile(stack, 99))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    return vmin, vmax


def overlay(rgb: np.ndarray, heatmap: np.ndarray, vmin: float, vmax: float, cmap_name: str = "jet") -> np.ndarray:
    norm = np.clip((heatmap - vmin) / (vmax - vmin), 0, 1)
    cmap = plt.get_cmap(cmap_name)
    color = (cmap(norm)[..., :3] * 255).astype(np.uint8)
    return np.uint8(0.55 * rgb + 0.45 * color)


def positive_focus_map(heatmap: np.ndarray) -> np.ndarray:
    """Keep regions whose occlusion reduces evidence for the explained class."""

    focused = np.asarray(heatmap, dtype=np.float32).copy()
    focused[focused < 0] = 0
    if float(np.nanmax(focused)) <= 0:
        focused = np.abs(np.asarray(heatmap, dtype=np.float32))
    return focused


def robust_single_map_limits(heatmap: np.ndarray) -> tuple[float, float]:
    focused = positive_focus_map(heatmap)
    vmin = 0.0
    vmax = float(np.nanpercentile(focused, 98))
    if not np.isfinite(vmax) or vmax <= 1e-8:
        vmax = float(np.nanmax(focused))
    if not np.isfinite(vmax) or vmax <= 1e-8:
        vmax = 1.0
    return vmin, vmax


def qualitative_overlay(rgb: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    focused = positive_focus_map(heatmap)
    vmin, vmax = robust_single_map_limits(focused)
    return overlay(rgb, focused, vmin, vmax, cmap_name="turbo")


def attention_region_name(heatmap: np.ndarray) -> str:
    heatmap = positive_focus_map(heatmap)
    h, w = heatmap.shape[:2]
    y, x = np.unravel_index(int(np.nanargmax(heatmap)), heatmap.shape)
    vertical = "upper" if y < h / 3 else "lower" if y >= 2 * h / 3 else "middle"
    horizontal = "left" if x < w / 3 else "right" if x >= 2 * w / 3 else "central"
    if vertical == "middle" and horizontal == "central":
        return "central face/image region"
    if vertical == "middle":
        return f"{horizontal} face/image region"
    if horizontal == "central":
        return f"{vertical} central face/image region"
    return f"{vertical}-{horizontal} face/image region"


def save_label_group_figure(
    output_path: Path,
    group_label: int,
    sample_results: list[dict[str, Any]],
) -> None:
    columns = ["Original image", "KNN", "Linear SVM", "CNN", "ViT"]
    n_rows = len(sample_results)
    fig, axes = plt.subplots(n_rows, len(columns), figsize=(18, max(3.5, 3.4 * n_rows)))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, sample in enumerate(sample_results):
        occlusion_maps = sample["occlusion_maps"]
        rgb = sample["rgb"]
        panels = [rgb] + [qualitative_overlay(rgb, occlusion_maps[name]) for name in MODEL_ORDER]
        for col_idx, image in enumerate(panels):
            ax = axes[row_idx, col_idx]
            ax.imshow(image)
            ax.axis("off")
            if col_idx == 0:
                sample_name = Path(sample["sample_path"]).name
                title = "Original image" if row_idx == 0 else ""
                ax.set_title(title, fontsize=10, fontweight="bold")
                ax.text(
                    0.5,
                    -0.08,
                    f"True: {LABEL_NAMES[group_label]}\n{sample_name}",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=8,
                )
            else:
                model_name = MODEL_ORDER[col_idx - 1]
                pred = sample["model_rows"][model_name]["predicted_label"]
                score = sample["model_rows"][model_name]["score"]
                correct = sample["model_rows"][model_name]["correct"]
                score_name = "margin" if model_name == "linear_svm" else "prob"
                mark = "correct" if correct else "wrong"
                column_title = columns[col_idx] if row_idx == 0 else columns[col_idx]
                ax.set_title(
                    f"{column_title}\nP={LABEL_NAMES[pred]} | {score_name}={score:.3f} | {mark}",
                    fontsize=8,
                    fontweight="bold" if row_idx == 0 else "normal",
                )

    fig.suptitle(
        f"Representative {LABEL_NAMES[group_label]} examples: cross-model occlusion heatmaps",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.93), h_pad=2.0, w_pad=1.0)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_comparison_figure(
    output_path: Path,
    rgb: np.ndarray,
    occlusion_maps: dict[str, np.ndarray],
    gradcam: np.ndarray,
    attention: np.ndarray,
    model_rows: dict[str, dict[str, Any]],
) -> None:
    vmin, vmax = normalise_for_display([occlusion_maps[name] for name in MODEL_ORDER])
    panels = [
        ("Original", rgb),
        ("KNN occlusion", overlay(rgb, occlusion_maps["knn"], vmin, vmax)),
        ("Linear SVM occlusion", overlay(rgb, occlusion_maps["linear_svm"], vmin, vmax)),
        ("CNN occlusion", overlay(rgb, occlusion_maps["cnn"], vmin, vmax)),
        ("ViT occlusion", overlay(rgb, occlusion_maps["vit"], vmin, vmax)),
        ("CNN Grad-CAM", overlay(rgb, gradcam, 0.0, 1.0)),
        ("ViT attention rollout", overlay(rgb, attention, 0.0, 1.0)),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(24, 4.8))
    for ax, (title, image) in zip(axes, panels):
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(title, fontsize=9)
    subtitle = []
    for name in MODEL_ORDER:
        row = model_rows[name]
        score_label = "prob" if name != "linear_svm" else "margin"
        subtitle.append(
            f"{name}: T={LABEL_NAMES[row['true_label']]} P={LABEL_NAMES[row['predicted_label']]} "
            f"{score_label}={row['score']:.3f} {row['category']}"
        )
    fig.suptitle("\n".join(subtitle), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_aggregate_maps(output_dir: Path, metadata_df: pd.DataFrame, model_name: str) -> None:
    for category in ["TP", "TN", "FP", "FN"]:
        rows = metadata_df[(metadata_df["model"] == model_name) & (metadata_df["category"] == category)]
        maps = []
        for path in rows["raw_occlusion_map_path"]:
            if isinstance(path, str) and Path(path).exists():
                maps.append(np.load(path))
        if not maps:
            continue
        mean_map = np.mean(np.stack(maps), axis=0)
        np.save(output_dir / f"{model_name}_{category}_mean_occlusion.npy", mean_map)
        plt.figure(figsize=(4, 4))
        plt.imshow(mean_map, cmap="jet")
        plt.axis("off")
        plt.title(f"{model_name} {category} mean occlusion")
        plt.tight_layout()
        plt.savefig(output_dir / f"{model_name}_{category}_mean_occlusion.png", dpi=200)
        plt.close()


def run_qualitative_examples(
    cfg: Config,
    dirs: dict[str, Path],
    scorers: dict[str, Callable[[np.ndarray], float]],
    predictors: dict[str, Callable[[np.ndarray], tuple[int, float]]],
) -> None:
    test_df = pd.read_csv(cfg.test_csv)
    required = {"sample_path", "label"}
    missing = required.difference(test_df.columns)
    if missing:
        raise ValueError(f"{cfg.test_csv} is missing required columns: {sorted(missing)}")
    test_df["label"] = test_df["label"].astype(int)
    selections = select_qualitative_examples(test_df, cfg)

    qualitative_dir = cfg.output_dir / "qualitative"
    qualitative_raw_dir = qualitative_dir / "raw_importance"
    qualitative_dir.mkdir(parents=True, exist_ok=True)
    qualitative_raw_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for group_label, group_df in selections.items():
        sample_results: list[dict[str, Any]] = []
        for sample_idx, row in tqdm(
            list(group_df.iterrows()),
            desc=f"Qualitative {LABEL_NAMES[group_label]}",
            dynamic_ncols=True,
        ):
            sample_path = str(row["sample_path"])
            rgb = load_rgb(sample_path, cfg.image_size)
            stem = f"{LABEL_NAMES[group_label].lower()}_{sample_idx:02d}_{safe_stem(sample_path)}"
            occlusion_maps: dict[str, np.ndarray] = {}
            model_rows: dict[str, dict[str, Any]] = {}

            for model_name in MODEL_ORDER:
                fill = (128, 128, 128)
                if model_name == "cnn":
                    fill = tuple(int(round(v * 255)) for v in IMAGENET_MEAN)
                pred, score = predictors[model_name](rgb)
                explained_score_fn = score_for_target_class(scorers[model_name], pred)
                grid, importance, original_score = occlusion_heatmap(
                    rgb,
                    explained_score_fn,
                    cfg.patch_size,
                    cfg.stride,
                    fill,
                )
                correct = pred == group_label
                attention_region = attention_region_name(importance) if correct else ""
                comment = (
                    f"Correct classification; strongest occlusion response is in the {attention_region}."
                    if correct
                    else "Incorrect classification; possible reasons include subtle artifacts, dataset/domain similarity, or model reliance on non-causal visual cues."
                )
                raw_path = qualitative_raw_dir / f"{stem}_{model_name}_occlusion.npy"
                grid_path = qualitative_raw_dir / f"{stem}_{model_name}_occlusion_grid.npy"
                np.save(raw_path, importance)
                np.save(grid_path, grid)
                model_row = {
                    "group": LABEL_NAMES[group_label],
                    "sample_path": sample_path,
                    "model": model_name,
                    "true_label": group_label,
                    "predicted_label": pred,
                    "predicted_label_name": LABEL_NAMES[pred],
                    "score": score,
                    "score_type": "decision_margin" if model_name == "linear_svm" else "probability_manipulated",
                    "explained_class": pred,
                    "explained_class_name": LABEL_NAMES[pred],
                    "correct": correct,
                    "attention_region_if_correct": attention_region,
                    "comment": comment,
                    "raw_occlusion_map_path": str(raw_path.resolve()),
                    "raw_occlusion_grid_path": str(grid_path.resolve()),
                    "original_explained_class_score": original_score,
                }
                rows.append(model_row)
                model_rows[model_name] = model_row
                occlusion_maps[model_name] = importance

            sample_results.append(
                {
                    "sample_path": sample_path,
                    "rgb": rgb,
                    "occlusion_maps": occlusion_maps,
                    "model_rows": model_rows,
                }
            )

        figure_name = "original_cross_model_heatmaps.png" if group_label == 0 else "manipulated_cross_model_heatmaps.png"
        save_label_group_figure(qualitative_dir / figure_name, group_label, sample_results)

    results_csv = qualitative_dir / "qualitative_cross_model_results.csv"
    pd.DataFrame(rows).to_csv(results_csv, index=False, encoding="utf-8-sig")

    summary = {
        "purpose": "Representative qualitative examples only; do not use these samples to claim new accuracy.",
        "selection": {
            "method": "random sample by true label from configured test CSV",
            "seed": cfg.seed,
            "per_label": cfg.qualitative_per_label,
            "test_csv": str(cfg.test_csv),
        },
        "figures": {
            "original": str((qualitative_dir / "original_cross_model_heatmaps.png").resolve()),
            "manipulated": str((qualitative_dir / "manipulated_cross_model_heatmaps.png").resolve()),
        },
        "results_csv": str(results_csv.resolve()),
        "note": "Heatmaps are occlusion sensitivity maps. For correctly classified models, the CSV reports the coarse region of strongest response. For incorrect predictions, comments are qualitative hypotheses only.",
    }
    (qualitative_dir / "README_qualitative_examples.md").write_text(
        "# Representative Cross-Model Heatmaps\n\n"
        "This folder contains qualitative, representative examples grouped by true label. "
        "The examples are selected with a fixed random seed from the test CSV and should not "
        "be used to report new accuracy.\n\n"
        "Generated figures:\n\n"
        "- `original_cross_model_heatmaps.png`\n"
        "- `manipulated_cross_model_heatmaps.png`\n\n"
        "Each row is one test sample and each column is: Original image, KNN, Linear SVM, CNN, ViT.\n\n"
        "See `qualitative_cross_model_results.csv` for predictions, scores, correctness, and brief comments.\n\n"
        f"```json\n{json.dumps(summary, indent=2)}\n```\n",
        encoding="utf-8",
    )
    print(f"Saved qualitative figures: {qualitative_dir}")
    print(f"Saved qualitative CSV: {results_csv}")


def write_readme(output_dir: Path, cfg: Config, inventory: dict[str, Any]) -> None:
    readme = f"""# Explainability Comparison

This directory contains spatial explanation outputs for the saved KNN-HOG,
Linear SVM-HOG, CNN, and ViT classifiers. The analysis loads existing model
artifacts only; it does not train, refit, recalibrate, or overwrite checkpoints.

## Run

Smoke test:

```powershell
python explainability_comparison.py --smoke --output_dir "{output_dir}"
```

Full selected-set run:

```powershell
python explainability_comparison.py --max_per_category {cfg.max_per_category} --patch_size {cfg.patch_size} --stride {cfg.stride} --output_dir "{output_dir}"
```

## Configuration Used

- Test CSV: `{cfg.test_csv}`
- Output directory: `{cfg.output_dir}`
- Image size: `{cfg.image_size}`
- Patch size / stride: `{cfg.patch_size}` / `{cfg.stride}`
- Seed: `{cfg.seed}`
- Device request: `{cfg.device}`
- Labels: `0 = Real`, `1 = Manipulated`

## Methods

Occlusion maps are model-agnostic. Each patch is replaced by a neutral pixel
value matched to the model family, then the Manipulated-class score is recomputed.
Importance is `original score - occluded score`. HOG models receive the occluded
image through grayscale resize, HOG extraction, fitted `StandardScaler`, then
the saved classifier.

CNN Grad-CAM uses the final convolutional layer of `SimpleCNN`. ViT uses
attention rollout and is reported as an attention-based visualisation, not a
complete causal explanation.

## Limitations

KNN uses probabilities, CNN and ViT use probabilities, while the Linear SVM uses
an uncalibrated decision margin because the saved `SGDClassifier(loss="hinge")`
does not provide probabilities. Occlusion maps can be sensitive to patch size,
stride, and neutral-fill choice. Attention rollout indicates propagated attention
patterns and should be interpreted cautiously.

## Inventory

```json
{json.dumps(inventory, indent=2)}
```
"""
    (output_dir / "README_explainability.md").write_text(readme, encoding="utf-8")


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    dirs = ensure_outputs(cfg.output_dir)
    device = resolve_device(cfg.device)

    inventory = {
        "knn_model": str(cfg.knn_model_path),
        "knn_scaler": str(cfg.knn_scaler_path),
        "linear_svm_model": str(cfg.linear_svm_model_path),
        "linear_svm_scaler": str(cfg.linear_svm_scaler_path),
        "cnn_checkpoint": str(cfg.cnn_checkpoint_path),
        "cnn_gradcam_layer": "SimpleCNN.features[-1][0]",
        "vit_model_dir": str(cfg.vit_model_dir),
        "vit_preprocessor_config": str((cfg.vit_model_dir / "preprocessor_config.json").resolve()),
        "vit_attention_method": "attention rollout over class-token attention",
        "label_mapping": LABEL_NAMES,
    }
    (cfg.output_dir / "inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")

    print("Loading saved artifacts...")
    knn = joblib.load(cfg.knn_model_path)
    knn_scaler = joblib.load(cfg.knn_scaler_path)
    linear_svm = joblib.load(cfg.linear_svm_model_path)
    linear_svm_scaler = joblib.load(cfg.linear_svm_scaler_path)

    cnn = SimpleCNN().to(device)
    checkpoint = torch.load(cfg.cnn_checkpoint_path, map_location=device)
    cnn.load_state_dict(checkpoint["model_state_dict"])
    cnn.eval()

    vit = AutoModelForImageClassification.from_pretrained(
        cfg.vit_model_dir,
        attn_implementation="eager",
    ).to(device)
    vit.eval()
    vit_mean, vit_std = read_vit_mean_std(cfg.vit_model_dir)

    scorers = {
        "knn": make_hog_scorer(knn, knn_scaler),
        "linear_svm": make_hog_scorer(linear_svm, linear_svm_scaler),
        "cnn": make_cnn_scorer(cnn, device, cfg.image_size),
        "vit": make_vit_scorer(vit, device, cfg.image_size, vit_mean, vit_std),
    }
    predictors = {
        "knn": make_hog_predictor(knn, knn_scaler),
        "linear_svm": make_hog_predictor(linear_svm, linear_svm_scaler),
        "cnn": make_cnn_predictor(cnn, device, cfg.image_size),
        "vit": make_vit_predictor(vit, device, cfg.image_size, vit_mean, vit_std),
    }

    if cfg.qualitative_examples:
        run_qualitative_examples(cfg, dirs, scorers, predictors)
        return

    cnn_pred = read_predictions(cfg.cnn_predictions_csv, "probability")
    vit_pred = read_predictions(cfg.vit_predictions_csv, "probability_manipulated")
    test_paths = set(pd.read_csv(cfg.test_csv)["sample_path"].astype(str))
    shared = select_shared_images(cnn_pred, cfg)
    shared = shared[shared["sample_path"].isin(test_paths)].reset_index(drop=True)
    if len(shared) == 0:
        raise RuntimeError("Selected images are not present in the configured test CSV.")
    shared.to_csv(cfg.output_dir / "shared_image_selection.csv", index=False, encoding="utf-8-sig")

    rows: list[dict[str, Any]] = []
    for image_idx, row in tqdm(list(shared.iterrows()), desc="Explaining images", dynamic_ncols=True):
        sample_path = str(row["sample_path"])
        true_label = int(row["true_label"])
        rgb = load_rgb(sample_path, cfg.image_size)
        stem = f"{image_idx:03d}_{safe_stem(sample_path)}"
        occlusion_maps: dict[str, np.ndarray] = {}
        model_rows: dict[str, dict[str, Any]] = {}

        for model_name in MODEL_ORDER:
            fill = (128, 128, 128)
            if model_name == "cnn":
                fill = tuple(int(round(v * 255)) for v in IMAGENET_MEAN)
            pred, score = predictors[model_name](rgb)
            grid, importance, original_score = occlusion_heatmap(
                rgb, scorers[model_name], cfg.patch_size, cfg.stride, fill
            )
            raw_path = dirs["raw"] / f"{stem}_{model_name}_occlusion.npy"
            grid_path = dirs["raw"] / f"{stem}_{model_name}_occlusion_grid.npy"
            np.save(raw_path, importance)
            np.save(grid_path, grid)

            fractions, deletion_scores, deletion_auc = deletion_curve(rgb, importance, scorers[model_name], fill)
            fidelity: dict[str, float] = {}
            for frac in cfg.deletion_fractions:
                masked_score = scorers[model_name](mask_top_fraction(rgb, importance, frac, fill))
                fidelity[f"drop_top_{int(frac * 100)}pct"] = float(original_score - masked_score)

            cat = confusion_category(true_label, pred)
            model_row = {
                "image_id": stem,
                "sample_path": sample_path,
                "model": model_name,
                "true_label": true_label,
                "predicted_label": pred,
                "score": score,
                "category": cat,
                "selection_category": row["selection_category"],
                "raw_occlusion_map_path": str(raw_path.resolve()),
                "raw_occlusion_grid_path": str(grid_path.resolve()),
                "deletion_fractions": json.dumps(fractions),
                "deletion_scores": json.dumps(deletion_scores),
                "deletion_auc": deletion_auc,
                **fidelity,
            }
            rows.append(model_row)
            model_rows[model_name] = model_row
            occlusion_maps[model_name] = importance

        gradcam_map = cnn_gradcam(cnn, rgb, device, cfg.image_size)
        attention_map = vit_attention_rollout(vit, rgb, device, cfg.image_size, vit_mean, vit_std)
        np.save(dirs["raw"] / f"{stem}_cnn_gradcam.npy", gradcam_map)
        np.save(dirs["raw"] / f"{stem}_vit_attention_rollout.npy", attention_map)
        save_comparison_figure(
            dirs["figures"] / f"{stem}_comparison.png",
            rgb,
            occlusion_maps,
            gradcam_map,
            attention_map,
            model_rows,
        )

    metadata = pd.DataFrame(rows)
    metadata_csv = cfg.output_dir / "explainability_results.csv"
    metadata.to_csv(metadata_csv, index=False, encoding="utf-8-sig")

    for model_name in MODEL_ORDER:
        save_aggregate_maps(dirs["aggregate"], metadata, model_name)

    write_readme(cfg.output_dir, cfg, inventory)
    print(f"Saved metadata: {metadata_csv}")
    print(f"Saved figures: {dirs['figures']}")
    print(f"Saved raw arrays: {dirs['raw']}")


if __name__ == "__main__":
    main()
