"""
First-stage dataset preparation for binary manipulated-media detection.

This script scans raw image/video datasets, infers Real/Manipulated labels,
preprocesses images, extracts video frames, and creates group-aware
train/validation/test splits without training any model.
"""

from __future__ import annotations

import argparse
import json
import random
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

REAL_TOP_LEVEL = {
    "celeba-hq resized (256x256)",
    "ffhq",
}

MANIPULATED_TOP_LEVEL = {
    "faceapp",
    "pggan_v1",
    "pggan_v2",
    "stargan",
    "stylegan_celeba",
    "stylegan_ffhq",
}

REAL_CELEB_DF_SUBFOLDERS = {
    "celeb-real",
    "youtube-real",
}

MANIPULATED_CELEB_DF_SUBFOLDERS = {
    "celeb-synthesis",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare raw media dataset metadata, frames, images, and group-aware splits."
    )
    parser.add_argument("--data_root", required=True, type=Path, help="Root folder containing all datasets.")
    parser.add_argument("--output_dir", default=Path("outputs/dataset"), type=Path, help="Output folder.")
    parser.add_argument("--image_size", default=224, type=int, help="Square image/frame size.")
    parser.add_argument("--frames_per_video", default=10, type=int, help="Frames to extract per video.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--max_images_per_source", default=None, type=int, help="Optional cap per source dataset.")
    parser.add_argument("--max_videos_per_source", default=None, type=int, help="Optional cap per source dataset.")
    return parser.parse_args()


def safe_name(value: str) -> str:
    """Create a filesystem-friendly folder/file component while keeping names readable."""
    return value.strip().replace("/", "_").replace("\\", "_").replace(":", "_")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def infer_label(path: Path, data_root: Path) -> tuple[int | None, str | None, str]:
    """
    Infer label from top-level dataset folder and known Celeb-DF-v2 subfolders.

    Returns (label, label_name, reason). label is None when uncertain.
    """
    try:
        relative_parts = path.relative_to(data_root).parts
    except ValueError:
        return None, None, "path_outside_data_root"

    if not relative_parts:
        return None, None, "empty_relative_path"

    top_level = relative_parts[0]
    top_level_norm = top_level.lower()

    if top_level_norm in REAL_TOP_LEVEL:
        return 0, "Real", f"top_level_real:{top_level}"

    if top_level_norm in MANIPULATED_TOP_LEVEL:
        return 1, "Manipulated", f"top_level_manipulated:{top_level}"

    if top_level_norm == "celeb-df-v2":
        lower_parts = [part.lower() for part in relative_parts[1:]]
        if any(part in REAL_CELEB_DF_SUBFOLDERS for part in lower_parts):
            return 0, "Real", "celeb_df_real_subfolder"
        if any(part in MANIPULATED_CELEB_DF_SUBFOLDERS for part in lower_parts):
            return 1, "Manipulated", "celeb_df_manipulated_subfolder"
        return None, None, "celeb_df_unknown_subfolder"

    return None, None, f"unrecognized_top_level:{top_level}"


def scan_files(
    data_root: Path,
    max_images_per_source: int | None = None,
    max_videos_per_source: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Recursively scan data_root and return usable metadata plus uncertain files."""
    usable_records: list[dict[str, Any]] = []
    uncertain_records: list[dict[str, Any]] = []
    scan_warnings: list[str] = []
    per_source_counts: dict[tuple[str, str], int] = defaultdict(int)

    all_files = [p for p in data_root.rglob("*") if p.is_file()]

    for file_path in tqdm(all_files, desc="Scanning files"):
        suffix = file_path.suffix.lower()
        if suffix not in IMAGE_EXTENSIONS and suffix not in VIDEO_EXTENSIONS:
            continue

        try:
            relative_parts = file_path.relative_to(data_root).parts
        except ValueError:
            relative_parts = file_path.parts

        source_dataset = relative_parts[0] if relative_parts else file_path.parent.name
        media_type = "image" if suffix in IMAGE_EXTENSIONS else "video"

        if media_type == "image" and max_images_per_source is not None:
            if per_source_counts[(source_dataset, "image")] >= max_images_per_source:
                continue

        if media_type == "video" and max_videos_per_source is not None:
            if per_source_counts[(source_dataset, "video")] >= max_videos_per_source:
                continue

        label, label_name, reason = infer_label(file_path, data_root)
        base_record = {
            "original_path": str(file_path.resolve()),
            "source_dataset": source_dataset,
            "media_type": media_type,
            "label": label,
            "label_name": label_name,
            "video_id": file_path.stem if media_type == "video" else "",
            "parent_folder": str(file_path.parent.resolve()),
            "file_name": file_path.name,
            "inference_reason": reason,
        }

        if label is None:
            uncertain_records.append(base_record)
            continue

        usable_records.append(base_record)
        per_source_counts[(source_dataset, media_type)] += 1

    if not usable_records:
        scan_warnings.append("No confidently labeled image or video files were found.")

    return pd.DataFrame(usable_records), pd.DataFrame(uncertain_records), scan_warnings


def process_images(
    raw_df: pd.DataFrame,
    output_dir: Path,
    image_size: int,
) -> tuple[pd.DataFrame, int, list[str]]:
    """Load, RGB-convert, resize, and save all usable image files."""
    image_rows = raw_df[raw_df["media_type"] == "image"].copy()
    processed_records: list[dict[str, Any]] = []
    failed_loads = 0
    process_warnings: list[str] = []

    for row in tqdm(image_rows.itertuples(index=False), total=len(image_rows), desc="Processing images"):
        original_path = Path(row.original_path)
        label_name = safe_name(row.label_name)
        source_dataset = safe_name(row.source_dataset)
        output_folder = output_dir / "processed_images" / "unsplit" / label_name / source_dataset
        output_folder.mkdir(parents=True, exist_ok=True)

        output_name = f"{original_path.stem}.jpg"
        processed_path = output_folder / output_name

        suffix_counter = 1
        while processed_path.exists():
            processed_path = output_folder / f"{original_path.stem}_{suffix_counter:04d}.jpg"
            suffix_counter += 1

        try:
            with Image.open(original_path) as img:
                img = img.convert("RGB")
                img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
                img.save(processed_path, format="JPEG", quality=95)
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            failed_loads += 1
            process_warnings.append(f"Failed image load: {original_path} ({exc})")
            continue

        processed_records.append(
            {
                "original_path": str(original_path.resolve()),
                "processed_path": str(processed_path.resolve()),
                "source_dataset": row.source_dataset,
                "media_type": "image",
                "label": int(row.label),
                "label_name": row.label_name,
                "video_id": "",
                "parent_folder": row.parent_folder,
                "file_name": row.file_name,
            }
        )

    return pd.DataFrame(processed_records), failed_loads, process_warnings


def get_even_frame_numbers(total_frames: int, frames_per_video: int) -> list[int]:
    if total_frames <= 0:
        return []
    if total_frames <= frames_per_video:
        return list(range(total_frames))
    return sorted(set(np.linspace(0, total_frames - 1, frames_per_video, dtype=int).tolist()))


def read_frame_at(cap: cv2.VideoCapture, frame_number: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = cap.read()
    if ok and frame is not None:
        return frame
    return None


def extract_video_frames(
    raw_df: pd.DataFrame,
    output_dir: Path,
    image_size: int,
    frames_per_video: int,
) -> tuple[pd.DataFrame, int, list[str]]:
    """Extract evenly spaced frames from usable videos."""
    video_rows = raw_df[raw_df["media_type"] == "video"].copy()
    frame_records: list[dict[str, Any]] = []
    failed_video_reads = 0
    video_warnings: list[str] = []

    for row in tqdm(video_rows.itertuples(index=False), total=len(video_rows), desc="Extracting video frames"):
        original_video_path = Path(row.original_path)
        label_name = safe_name(row.label_name)
        source_dataset = safe_name(row.source_dataset)
        video_stem = safe_name(original_video_path.stem)
        video_id = f"{row.source_dataset}/{original_video_path.stem}"

        frame_folder = (
            output_dir
            / "extracted_frames"
            / "unsplit"
            / label_name
            / source_dataset
            / video_stem
        )
        frame_folder.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(original_video_path))
        if not cap.isOpened():
            failed_video_reads += 1
            video_warnings.append(f"Failed video open: {original_video_path}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        candidate_numbers = get_even_frame_numbers(total_frames, frames_per_video)
        extracted = 0

        for frame_index, frame_number in enumerate(candidate_numbers):
            frame = read_frame_at(cap, frame_number)
            if frame is None:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
            frame_path = frame_folder / f"frame_{extracted:03d}.jpg"
            cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            frame_records.append(
                {
                    "original_video_path": str(original_video_path.resolve()),
                    "frame_path": str(frame_path.resolve()),
                    "source_dataset": row.source_dataset,
                    "media_type": "video_frame",
                    "label": int(row.label),
                    "label_name": row.label_name,
                    "video_id": video_id,
                    "frame_index": extracted,
                    "frame_number": int(frame_number),
                }
            )
            extracted += 1

        cap.release()

        if extracted == 0:
            failed_video_reads += 1
            video_warnings.append(f"No readable frames extracted: {original_video_path}")
        elif extracted < frames_per_video:
            video_warnings.append(
                f"Only extracted {extracted}/{frames_per_video} readable frames: {original_video_path}"
            )

    return pd.DataFrame(frame_records), failed_video_reads, video_warnings


def build_dataset_index(processed_images_df: pd.DataFrame, frames_df: pd.DataFrame) -> pd.DataFrame:
    """Create unified sample-level dataset index."""
    index_records: list[dict[str, Any]] = []

    for row in processed_images_df.itertuples(index=False):
        index_records.append(
            {
                "sample_path": row.processed_path,
                "source_dataset": row.source_dataset,
                "media_type": "image",
                "label": int(row.label),
                "label_name": row.label_name,
                "group_id": row.original_path,
            }
        )

    for row in frames_df.itertuples(index=False):
        index_records.append(
            {
                "sample_path": row.frame_path,
                "source_dataset": row.source_dataset,
                "media_type": "video_frame",
                "label": int(row.label),
                "label_name": row.label_name,
                "group_id": row.video_id,
            }
        )

    return pd.DataFrame(index_records)


def split_score(df: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> float:
    """Lower is better: penalize class-ratio drift and split-size drift."""
    if len(df) == 0:
        return 0.0

    overall = df["label"].value_counts(normalize=True).to_dict()
    target_sizes = {"train": 0.70, "val": 0.15, "test": 0.15}
    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    score = 0.0

    for split_name, idx in splits.items():
        split_df = df.iloc[idx]
        actual_size = len(split_df) / len(df)
        score += abs(actual_size - target_sizes[split_name]) * 2.0

        split_dist = split_df["label"].value_counts(normalize=True).to_dict()
        for label in [0, 1]:
            score += abs(split_dist.get(label, 0.0) - overall.get(label, 0.0))

    return score


def attempt_group_split(df: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Create one 70/15/15 group-aware split attempt using GroupShuffleSplit."""
    if len(df) < 3 or df["group_id"].nunique() < 3:
        return None

    groups = df["group_id"].to_numpy()
    all_indices = np.arange(len(df))

    try:
        train_splitter = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=seed)
        train_idx, temp_idx = next(train_splitter.split(df, groups=groups))

        temp_df = df.iloc[temp_idx].reset_index(drop=False).rename(columns={"index": "original_index"})
        if temp_df["group_id"].nunique() < 2:
            return None

        temp_splitter = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=seed + 1)
        val_local_idx, test_local_idx = next(temp_splitter.split(temp_df, groups=temp_df["group_id"].to_numpy()))

        val_idx = temp_df.iloc[val_local_idx]["original_index"].to_numpy()
        test_idx = temp_df.iloc[test_local_idx]["original_index"].to_numpy()

        return all_indices[train_idx], val_idx, test_idx
    except ValueError:
        return None


def fallback_group_split(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """
    Deterministic group assignment fallback for tiny datasets.

    This preserves group integrity but may not preserve class balance well.
    """
    rng = random.Random(seed)
    group_ids = sorted(df["group_id"].unique().tolist())
    rng.shuffle(group_ids)

    train_cut = max(1, int(round(len(group_ids) * 0.70)))
    val_cut = max(train_cut + 1, int(round(len(group_ids) * 0.85))) if len(group_ids) > 2 else train_cut

    train_groups = set(group_ids[:train_cut])
    val_groups = set(group_ids[train_cut:val_cut])
    test_groups = set(group_ids[val_cut:])

    train_df = df[df["group_id"].isin(train_groups)].copy()
    val_df = df[df["group_id"].isin(val_groups)].copy()
    test_df = df[df["group_id"].isin(test_groups)].copy()
    return train_df, val_df, test_df, "fallback_tiny_group_split"


def create_group_split(
    dataset_index_df: pd.DataFrame,
    seed: int,
    attempts: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Create train/val/test CSV data with group-aware splitting."""
    df = dataset_index_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if len(df) == 0:
        empty = df.copy()
        return empty, empty, empty, "empty_dataset"

    best: tuple[float, np.ndarray, np.ndarray, np.ndarray] | None = None
    for offset in range(attempts):
        split = attempt_group_split(df, seed + offset)
        if split is None:
            continue
        train_idx, val_idx, test_idx = split
        score = split_score(df, train_idx, val_idx, test_idx)
        if best is None or score < best[0]:
            best = (score, train_idx, val_idx, test_idx)

    if best is None:
        return fallback_group_split(df, seed)

    _, train_idx, val_idx, test_idx = best
    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = df.iloc[val_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)
    return train_df, val_df, test_df, "group_shuffle_best_of_attempts"


def count_by_label(df: pd.DataFrame) -> dict[str, int]:
    if len(df) == 0 or "label_name" not in df:
        return {}
    return {str(k): int(v) for k, v in df["label_name"].value_counts().to_dict().items()}


def save_csv(df: pd.DataFrame, path: Path, columns: list[str] | None = None) -> None:
    if columns is not None:
        for column in columns:
            if column not in df.columns:
                df[column] = pd.Series(dtype="object")
        df = df[columns]
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_summary(
    output_dir: Path,
    raw_df: pd.DataFrame,
    frames_df: pd.DataFrame,
    dataset_index_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    uncertain_df: pd.DataFrame,
    warnings_list: list[str],
    failed_image_loads: int,
    failed_video_reads: int,
    split_method: str,
) -> dict[str, Any]:
    """Save dataset_summary.json and return the summary object."""
    summary = {
        "total_raw_images": int((raw_df["media_type"] == "image").sum()) if len(raw_df) else 0,
        "total_videos": int((raw_df["media_type"] == "video").sum()) if len(raw_df) else 0,
        "total_extracted_frames": int(len(frames_df)),
        "final_number_of_samples": int(len(dataset_index_df)),
        "class_counts_overall": count_by_label(dataset_index_df),
        "class_counts_by_split": {
            "train": count_by_label(train_df),
            "val": count_by_label(val_df),
            "test": count_by_label(test_df),
        },
        "source_dataset_counts": {
            str(k): int(v) for k, v in dataset_index_df["source_dataset"].value_counts().to_dict().items()
        }
        if len(dataset_index_df)
        else {},
        "media_type_counts": {
            str(k): int(v) for k, v in dataset_index_df["media_type"].value_counts().to_dict().items()
        }
        if len(dataset_index_df)
        else {},
        "uncertain_file_count": int(len(uncertain_df)),
        "failed_image_loads": int(failed_image_loads),
        "failed_video_reads": int(failed_video_reads),
        "split_method": split_method,
        "warnings": warnings_list,
    }

    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def print_summary(
    summary: dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    print("\nDataset preparation complete")
    print("=" * 36)
    print(f"Total usable samples: {summary['final_number_of_samples']}")
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples:   {len(val_df)}")
    print(f"Test samples:  {len(test_df)}")
    print()

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        counts = count_by_label(split_df)
        print(
            f"{split_name:>5}: "
            f"Real={counts.get('Real', 0)} | Manipulated={counts.get('Manipulated', 0)}"
        )

    print()
    print(f"Uncertain files:        {summary['uncertain_file_count']}")
    print(f"Failed image loads:     {summary['failed_image_loads']}")
    print(f"Failed video reads:     {summary['failed_video_reads']}")
    print(f"Extracted video frames: {summary['total_extracted_frames']}")
    print(f"Split method:           {summary['split_method']}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_root.exists():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")

    warnings_list: list[str] = []
    raw_df, uncertain_df, scan_warnings = scan_files(
        data_root=data_root,
        max_images_per_source=args.max_images_per_source,
        max_videos_per_source=args.max_videos_per_source,
    )
    warnings_list.extend(scan_warnings)

    processed_images_df, failed_image_loads, image_warnings = process_images(
        raw_df=raw_df,
        output_dir=output_dir,
        image_size=args.image_size,
    )
    warnings_list.extend(image_warnings)

    if "processed_path" not in raw_df.columns:
        raw_df["processed_path"] = ""
    if len(processed_images_df):
        processed_path_map = dict(
            zip(processed_images_df["original_path"], processed_images_df["processed_path"])
        )
        raw_df["processed_path"] = raw_df["original_path"].map(processed_path_map).fillna("")

    frames_df, failed_video_reads, video_warnings = extract_video_frames(
        raw_df=raw_df,
        output_dir=output_dir,
        image_size=args.image_size,
        frames_per_video=args.frames_per_video,
    )
    warnings_list.extend(video_warnings)

    dataset_index_df = build_dataset_index(processed_images_df, frames_df)
    train_df, val_df, test_df, split_method = create_group_split(dataset_index_df, args.seed)

    metadata_raw_columns = [
        "original_path",
        "source_dataset",
        "media_type",
        "label",
        "label_name",
        "video_id",
        "parent_folder",
        "file_name",
        "processed_path",
    ]
    metadata_frames_columns = [
        "original_video_path",
        "frame_path",
        "source_dataset",
        "media_type",
        "label",
        "label_name",
        "video_id",
        "frame_index",
        "frame_number",
    ]
    dataset_index_columns = [
        "sample_path",
        "source_dataset",
        "media_type",
        "label",
        "label_name",
        "group_id",
    ]

    save_csv(raw_df, output_dir / "metadata_raw.csv", metadata_raw_columns)
    save_csv(frames_df, output_dir / "metadata_frames.csv", metadata_frames_columns)
    save_csv(uncertain_df, output_dir / "uncertain_files.csv")
    save_csv(dataset_index_df, output_dir / "dataset_index.csv", dataset_index_columns)
    save_csv(train_df, output_dir / "train.csv", dataset_index_columns)
    save_csv(val_df, output_dir / "val.csv", dataset_index_columns)
    save_csv(test_df, output_dir / "test.csv", dataset_index_columns)

    summary = save_summary(
        output_dir=output_dir,
        raw_df=raw_df,
        frames_df=frames_df,
        dataset_index_df=dataset_index_df,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        uncertain_df=uncertain_df,
        warnings_list=warnings_list,
        failed_image_loads=failed_image_loads,
        failed_video_reads=failed_video_reads,
        split_method=split_method,
    )

    if warnings_list:
        warnings.warn(
            f"Completed with {len(warnings_list)} warnings. See dataset_summary.json for details.",
            RuntimeWarning,
        )

    print_summary(summary, train_df, val_df, test_df)


if __name__ == "__main__":
    main()
