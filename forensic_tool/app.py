"""
Local ViT-based media manipulation forensics tool.

Run:
    python forensic_tool/app.py

Then open:
    http://127.0.0.1:7860
"""

from __future__ import annotations

import cgi
import html
import json
import os
import random
import shutil
import socket
import sys
import time
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from transformers import AutoModelForImageClassification


ROOT_DIR = Path(__file__).resolve().parents[1]
TOOL_DIR = Path(__file__).resolve().parent
STATIC_DIR = TOOL_DIR / "static"
RESULTS_DIR = STATIC_DIR / "results"
MODEL_DIR = ROOT_DIR / "outputs" / "vit" / "models" / "vit_best_model"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
LABELS = {0: "Real", 1: "Manipulated"}


@dataclass
class Prediction:
    label: int
    probability: float

    @property
    def label_name(self) -> str:
        return LABELS[self.label]

    @property
    def risk_class(self) -> str:
        if self.probability >= 0.85:
            return "high"
        if self.probability >= 0.5:
            return "medium"
        return "low"


class ViTDetector:
    def __init__(self, model_dir: Path) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForImageClassification.from_pretrained(
            model_dir,
            attn_implementation="eager",
        ).to(self.device)
        self.model.eval()
        self.mean, self.std, self.image_size = self._load_preprocessor(model_dir)
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.image_size + 32),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

    @staticmethod
    def _load_preprocessor(model_dir: Path) -> tuple[list[float], list[float], int]:
        path = model_dir / "preprocessor_config.json"
        if not path.exists():
            return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], 224
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        size = data.get("size", {})
        image_size = int(size.get("height") or size.get("width") or 224)
        return list(data.get("image_mean", [0.5, 0.5, 0.5])), list(data.get("image_std", [0.5, 0.5, 0.5])), image_size

    def preprocess_pil(self, image: Image.Image) -> torch.Tensor:
        return self.transform(ImageOps.exif_transpose(image).convert("RGB")).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict_pil(self, image: Image.Image) -> Prediction:
        tensor = self.preprocess_pil(image)
        logits = self.model(pixel_values=tensor).logits
        probability = float(torch.softmax(logits, dim=1)[0, 1].detach().cpu().item())
        return Prediction(label=int(probability >= 0.5), probability=probability)

    def attention_rollout(self, image: Image.Image) -> np.ndarray:
        tensor = self.preprocess_pil(image)
        with torch.no_grad():
            outputs = self.model(pixel_values=tensor, output_attentions=True)
        attentions = outputs.attentions
        if attentions is None:
            raise RuntimeError("The ViT model did not return attention tensors.")

        rollout = torch.eye(attentions[0].shape[-1], device=self.device)
        for attention in attentions:
            attn = attention[0].mean(dim=0)
            attn = attn + torch.eye(attn.shape[0], device=self.device)
            attn = attn / attn.sum(dim=-1, keepdim=True)
            rollout = attn @ rollout

        cls_attention = rollout[0, 1:].detach().cpu().numpy()
        side = int(np.sqrt(cls_attention.shape[0]))
        heatmap = cls_attention[: side * side].reshape(side, side)
        heatmap = heatmap - heatmap.min()
        denom = heatmap.max()
        if denom > 0:
            heatmap = heatmap / denom
        return heatmap.astype(np.float32)


DETECTOR: ViTDetector | None = None


def get_detector() -> ViTDetector:
    global DETECTOR
    if DETECTOR is None:
        DETECTOR = ViTDetector(MODEL_DIR)
    return DETECTOR


def safe_filename(filename: str) -> str:
    name = Path(filename).name.replace(" ", "_")
    keep = [c for c in name if c.isalnum() or c in "._-()"]
    cleaned = "".join(keep).strip(".")
    return cleaned or "upload.bin"


def prepare_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def image_to_display_rgb(image: Image.Image, image_size: int) -> np.ndarray:
    img = ImageOps.exif_transpose(image).convert("RGB")
    img = img.resize((image_size, image_size), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def heatmap_overlay(rgb: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    heatmap = cv2.resize(heatmap, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    heatmap = np.clip(heatmap, 0.0, 1.0)
    color = (plt.get_cmap("jet")(heatmap)[..., :3] * 255).astype(np.uint8)
    return np.uint8(0.58 * rgb + 0.42 * color)


def suspicious_boxes(heatmap: np.ndarray, width: int, height: int) -> list[tuple[int, int, int, int]]:
    heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)
    threshold = max(0.55, float(np.quantile(heatmap, 0.88)))
    mask = (heatmap >= threshold).astype(np.uint8) * 255
    kernel = np.ones((7, 7), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    min_area = max(80, int(width * height * 0.01))
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h >= min_area:
            boxes.append((x, y, w, h))
    boxes.sort(key=lambda box: box[2] * box[3], reverse=True)
    return boxes[:3]


def draw_boxes(rgb: np.ndarray, boxes: list[tuple[int, int, int, int]], probability: float) -> np.ndarray:
    annotated = rgb.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 45, 75), 2)
    text = f"Manipulation score: {probability:.2%}"
    cv2.putText(annotated, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(annotated, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 45, 75), 1, cv2.LINE_AA)
    return annotated


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    Image.fromarray(rgb).save(path)


def analyse_image(upload_path: Path, session_dir: Path) -> dict[str, Any]:
    detector = get_detector()
    with Image.open(upload_path) as img:
        prediction = detector.predict_pil(img)
        rgb = image_to_display_rgb(img, detector.image_size)
        heatmap = detector.attention_rollout(img)

    overlay = heatmap_overlay(rgb, heatmap)
    boxes = suspicious_boxes(heatmap, rgb.shape[1], rgb.shape[0]) if prediction.label == 1 else []
    annotated = draw_boxes(overlay if prediction.label == 1 else rgb, boxes, prediction.probability)

    original_path = session_dir / "original.png"
    overlay_path = session_dir / "attention_overlay.png"
    annotated_path = session_dir / "annotated.png"
    heatmap_path = session_dir / "attention_rollout.npy"
    save_rgb(original_path, rgb)
    save_rgb(overlay_path, overlay)
    save_rgb(annotated_path, annotated)
    np.save(heatmap_path, cv2.resize(heatmap, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC))

    return {
        "kind": "image",
        "prediction": prediction,
        "original": original_path,
        "overlay": overlay_path,
        "annotated": annotated_path,
        "heatmap": heatmap_path,
        "boxes": boxes,
    }


def extract_video_frames(video_path: Path, session_dir: Path, sample_every_sec: float, max_frames: int) -> list[dict[str, Any]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if fps > 0 and total_frames > 0 else 0.0
    step = max(1, int(round(fps * sample_every_sec)))

    frames: list[dict[str, Any]] = []
    frame_index = 0
    sampled = 0
    while sampled < max_frames:
        ok = cap.grab()
        if not ok:
            break
        if frame_index % step == 0:
            ok, bgr = cap.retrieve()
            if ok:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frame_path = session_dir / f"frame_{sampled:04d}.jpg"
                save_rgb(frame_path, rgb)
                frames.append(
                    {
                        "frame_index": frame_index,
                        "time_sec": frame_index / fps if fps > 0 else float(sampled),
                        "path": frame_path,
                        "fps": fps,
                        "duration": duration,
                    }
                )
                sampled += 1
        frame_index += 1
    cap.release()
    return frames


def analyse_video(upload_path: Path, session_dir: Path, sample_every_sec: float = 1.0, max_frames: int = 32) -> dict[str, Any]:
    detector = get_detector()
    frames = extract_video_frames(upload_path, session_dir, sample_every_sec, max_frames)
    if not frames:
        raise RuntimeError("No frames could be extracted from the uploaded video.")

    analysed: list[dict[str, Any]] = []
    for item in frames:
        with Image.open(item["path"]) as img:
            prediction = detector.predict_pil(img)
        item["prediction"] = prediction
        analysed.append(item)

    suspicious = [item for item in analysed if item["prediction"].probability >= 0.5]
    top_frames = sorted(analysed, key=lambda x: x["prediction"].probability, reverse=True)[: min(5, len(analysed))]

    for rank, item in enumerate(top_frames):
        with Image.open(item["path"]) as img:
            prediction = item["prediction"]
            rgb = image_to_display_rgb(img, detector.image_size)
            heatmap = detector.attention_rollout(img)
            overlay = heatmap_overlay(rgb, heatmap)
            boxes = suspicious_boxes(heatmap, rgb.shape[1], rgb.shape[0]) if prediction.label == 1 else []
            annotated = draw_boxes(overlay if prediction.label == 1 else rgb, boxes, prediction.probability)
        annotated_path = session_dir / f"suspicious_{rank:02d}_annotated.png"
        overlay_path = session_dir / f"suspicious_{rank:02d}_overlay.png"
        npy_path = session_dir / f"suspicious_{rank:02d}_attention.npy"
        save_rgb(annotated_path, annotated)
        save_rgb(overlay_path, overlay)
        np.save(npy_path, cv2.resize(heatmap, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC))
        item["annotated"] = annotated_path
        item["overlay"] = overlay_path
        item["heatmap"] = npy_path
        item["boxes"] = boxes

    max_probability = max(item["prediction"].probability for item in analysed)
    mean_top_probability = float(np.mean([item["prediction"].probability for item in top_frames]))
    overall = Prediction(label=int(max_probability >= 0.5), probability=float(max_probability))

    report_rows = []
    for item in analysed:
        report_rows.append(
            {
                "frame_index": item["frame_index"],
                "time_sec": round(item["time_sec"], 3),
                "probability_manipulated": item["prediction"].probability,
                "predicted_label": item["prediction"].label,
            }
        )
    with (session_dir / "frame_report.csv").open("w", encoding="utf-8") as f:
        f.write("frame_index,time_sec,probability_manipulated,predicted_label\n")
        for row in report_rows:
            f.write(f"{row['frame_index']},{row['time_sec']},{row['probability_manipulated']:.8f},{row['predicted_label']}\n")

    return {
        "kind": "video",
        "prediction": overall,
        "frames": analysed,
        "top_frames": top_frames,
        "suspicious_count": len(suspicious),
        "mean_top_probability": mean_top_probability,
        "report": session_dir / "frame_report.csv",
    }


def rel_static(path: Path) -> str:
    return "/" + path.relative_to(TOOL_DIR).as_posix()


def render_home(message: str = "") -> str:
    escaped = html.escape(message)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Media Manipulation Forensics</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  <main class="shell">
    <section class="topbar">
      <div>
        <p class="eyebrow">Digital Forensics Assistant</p>
        <h1>Media Manipulation Detector</h1>
      </div>
      <div class="status-pill">ViT model loaded on demand</div>
    </section>

    <form id="scan-form" class="scanner" action="/analyze" method="post" enctype="multipart/form-data">
      <fieldset class="frame-panel">
        <legend>Frame</legend>
        <div id="drop-zone" class="drop-zone">
          <input id="media" name="media" type="file" accept="image/*,video/*" required>
          <label for="media">
            <span id="upload-title" class="upload-title">Upload image/video</span>
            <span id="upload-subtitle" class="upload-subtitle">JPEG, PNG, MP4, AVI, MOV, MKV</span>
          </label>
        </div>
        <div class="settings-grid">
          <label>Video sample interval
            <input name="sample_every_sec" type="number" value="1.0" min="0.2" max="10" step="0.2">
          </label>
          <label>Max sampled frames
            <input name="max_frames" type="number" value="32" min="4" max="120" step="1">
          </label>
        </div>
        <button id="analyse-button" type="submit" disabled>Analyse Media</button>
      </fieldset>

      <fieldset id="detecting-panel" class="frame-panel muted-panel">
        <legend>Detecting</legend>
        <div class="analysis-placeholder">
          <span id="analysis-status">Waiting for upload</span>
          <div class="progress"><i></i></div>
        </div>
      </fieldset>
    </form>
    <p class="notice">{escaped}</p>
  </main>
  <script>
    const form = document.getElementById("scan-form");
    const media = document.getElementById("media");
    const dropZone = document.getElementById("drop-zone");
    const uploadTitle = document.getElementById("upload-title");
    const uploadSubtitle = document.getElementById("upload-subtitle");
    const analyseButton = document.getElementById("analyse-button");
    const detectingPanel = document.getElementById("detecting-panel");
    const analysisStatus = document.getElementById("analysis-status");

    media.addEventListener("change", () => {{
      const file = media.files && media.files[0];
      if (!file) {{
        dropZone.classList.remove("ready");
        uploadTitle.textContent = "Upload image/video";
        uploadSubtitle.textContent = "JPEG, PNG, MP4, AVI, MOV, MKV";
        analyseButton.disabled = true;
        analysisStatus.textContent = "Waiting for upload";
        return;
      }}
      dropZone.classList.add("ready");
      uploadTitle.textContent = "Uploaded";
      uploadSubtitle.textContent = file.name;
      analyseButton.disabled = false;
      analysisStatus.textContent = "Upload complete";
      detectingPanel.classList.remove("analysing");
    }});

    form.addEventListener("submit", () => {{
      uploadTitle.textContent = "Uploading";
      analysisStatus.textContent = "Analysing";
      detectingPanel.classList.add("analysing");
      analyseButton.disabled = true;
      analyseButton.textContent = "Analysing...";
    }});
  </script>
</body>
</html>"""


def render_result(result: dict[str, Any], session_dir: Path) -> str:
    prediction: Prediction = result["prediction"]
    risk = prediction.risk_class
    if result["kind"] == "image":
        return render_image_result(result, risk)
    return render_video_result(result, risk)


def result_header(prediction: Prediction, kind: str) -> str:
    verdict = "Manipulated" if prediction.label == 1 else "Real"
    return f"""
    <section class="result-header">
      <a href="/" class="back-link">New scan</a>
      <div>
        <p class="eyebrow">{html.escape(kind.title())} Analysis</p>
        <h1>{verdict}</h1>
      </div>
      <div class="score-card {prediction.risk_class}">
        <span>Manipulation score</span>
        <strong>{prediction.probability:.1%}</strong>
      </div>
    </section>
    """


def render_image_result(result: dict[str, Any], risk: str) -> str:
    prediction: Prediction = result["prediction"]
    note = "Highlighted regions show the ViT attention rollout associated with the Manipulated prediction."
    if prediction.label == 0:
        note = "No manipulation was predicted; the original image is shown with the model score."
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Image Result</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  <main class="shell">
    {result_header(prediction, "image")}
    <section class="result-grid">
      <figure class="media-panel">
        <figcaption>Original</figcaption>
        <img src="{rel_static(result['original'])}" alt="Original upload">
      </figure>
      <figure class="media-panel">
        <figcaption>Detection Overlay</figcaption>
        <img src="{rel_static(result['annotated'])}" alt="Annotated detection">
      </figure>
      <aside class="details-panel {risk}">
        <h2>Detection</h2>
        <p class="verdict">{prediction.label_name}</p>
        <p>Manipulation: <strong>{prediction.probability:.2%}</strong></p>
        <p>Regions found: <strong>{len(result['boxes'])}</strong></p>
        <p>{html.escape(note)}</p>
      </aside>
    </section>
  </main>
</body>
</html>"""


def render_video_result(result: dict[str, Any], risk: str) -> str:
    prediction: Prediction = result["prediction"]
    frames = result["frames"]
    top = result["top_frames"]
    duration = frames[0].get("duration", 0.0)
    timeline = []
    for item in frames:
        prob = item["prediction"].probability
        left = 0.0 if duration <= 0 else min(100.0, 100.0 * item["time_sec"] / max(duration, 0.001))
        css = "hot" if prob >= 0.5 else "cool"
        timeline.append(f'<span class="{css}" style="left:{left:.2f}%" title="{item["time_sec"]:.2f}s | {prob:.1%}"></span>')
    cards = []
    for item in top:
        annotated = item.get("annotated") or item["path"]
        cards.append(
            f"""
            <article class="frame-card">
              <img src="{rel_static(annotated)}" alt="Suspicious frame">
              <div>
                <strong>{item['time_sec']:.2f}s</strong>
                <span>Frame {item['frame_index']}</span>
                <span>{item['prediction'].probability:.1%}</span>
              </div>
            </article>
            """
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Video Result</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  <main class="shell">
    {result_header(prediction, "video")}
    <section class="video-layout">
      <div class="timeline-panel">
        <div class="timeline">{''.join(timeline)}</div>
        <div class="timeline-labels">
          <span>0s</span>
          <span>{duration:.1f}s</span>
        </div>
      </div>
      <aside class="details-panel {risk}">
        <h2>Detection</h2>
        <p class="verdict">{prediction.label_name}</p>
        <p>Peak manipulation: <strong>{prediction.probability:.2%}</strong></p>
        <p>Suspicious sampled frames: <strong>{result['suspicious_count']}</strong></p>
        <p>Frame report: <a href="{rel_static(result['report'])}">CSV</a></p>
      </aside>
    </section>
    <section class="frame-list">
      <h2>Most Suspicious Frames</h2>
      <div class="frame-grid">{''.join(cards)}</div>
    </section>
  </main>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.respond_html(render_home())
            return
        if parsed.path.startswith("/static/"):
            self.serve_static(parsed.path)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/analyze":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            result_html = self.handle_upload()
            self.respond_html(result_html)
        except Exception as exc:
            self.respond_html(render_home(f"Analysis failed: {exc}"), status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def handle_upload(self) -> str:
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
            },
        )
        file_item = form["media"] if "media" in form else None
        if file_item is None or not getattr(file_item, "filename", ""):
            raise ValueError("No media file was uploaded.")

        session_dir = RESULTS_DIR / f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
        session_dir.mkdir(parents=True, exist_ok=True)
        filename = safe_filename(file_item.filename)
        upload_path = session_dir / filename
        with upload_path.open("wb") as f:
            shutil.copyfileobj(file_item.file, f)

        suffix = upload_path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            result = analyse_image(upload_path, session_dir)
        elif suffix in VIDEO_EXTENSIONS:
            sample_every = float(form.getfirst("sample_every_sec", "1.0"))
            max_frames = int(form.getfirst("max_frames", "32"))
            result = analyse_video(upload_path, session_dir, sample_every, max_frames)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        result_payload = serialise_result(result)
        with (session_dir / "result.json").open("w", encoding="utf-8") as f:
            json.dump(result_payload, f, indent=2, ensure_ascii=False)
        return render_result(result, session_dir)

    @staticmethod
    def translate_static_path(path: str) -> Path:
        relative = unquote(path.removeprefix("/static/"))
        candidate = (STATIC_DIR / relative).resolve()
        if STATIC_DIR.resolve() not in candidate.parents and candidate != STATIC_DIR.resolve():
            raise ValueError("Invalid static path.")
        return candidate

    def serve_static(self, path: str) -> None:
        try:
            candidate = self.translate_static_path(path)
        except ValueError:
            self.send_error(HTTPStatus.BAD_REQUEST)
            return
        if not candidate.exists() or not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        mime = "application/octet-stream"
        if candidate.suffix.lower() in {".css"}:
            mime = "text/css; charset=utf-8"
        elif candidate.suffix.lower() in {".png"}:
            mime = "image/png"
        elif candidate.suffix.lower() in {".jpg", ".jpeg"}:
            mime = "image/jpeg"
        elif candidate.suffix.lower() in {".csv"}:
            mime = "text/csv; charset=utf-8"
        elif candidate.suffix.lower() in {".json"}:
            mime = "application/json; charset=utf-8"
        data = candidate.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def respond_html(self, body: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write(f"[forensic-tool] {self.address_string()} - {format % args}\n")


def serialise_result(result: dict[str, Any]) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, Prediction):
            return {"label": value.label, "label_name": value.label_name, "probability": value.probability}
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, list):
            return [convert(x) for x in value]
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        return value

    return convert(result)


def main() -> None:
    prepare_dirs()
    random.seed(42)
    np.random.seed(42)
    port = int(os.environ.get("FORENSIC_TOOL_PORT", "7860"))
    host = os.environ.get("FORENSIC_TOOL_HOST", "0.0.0.0")
    server = ThreadingHTTPServer((host, port), Handler)
    lan_ip = get_lan_ip()
    print(f"Media Manipulation Forensics Tool")
    print(f"Model: {MODEL_DIR}")
    print(f"Open: http://127.0.0.1:{port}")
    print(f"LAN:  http://{lan_ip}:{port}")
    server.serve_forever()


def get_lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())


if __name__ == "__main__":
    main()
