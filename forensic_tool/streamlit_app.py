from __future__ import annotations

import io
import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from transformers import AutoModelForImageClassification, AutoImageProcessor

LABELS = {0: "Real", 1: "Manipulated"}
IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]
VIDEO_TYPES = ["mp4", "avi", "mov", "mkv", "webm"]

DEFAULT_MODEL_SOURCE = "Penny1507288/deepfake_detection"

st.set_page_config(
    page_title="Media Manipulation Forensics",
    page_icon="🔎",
    layout="wide",
)

def get_model_source() -> str:
    # Priority: Streamlit secret -> environment variable -> local checkpoint.
    if "MODEL_SOURCE" in st.secrets:
        return str(st.secrets["MODEL_SOURCE"])
    if os.environ.get("FORENSIC_MODEL_SOURCE"):
        return os.environ["FORENSIC_MODEL_SOURCE"]
    return DEFAULT_MODEL_SOURCE

@st.cache_resource(show_spinner="Loading ViT model...")
def load_detector(model_source: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForImageClassification.from_pretrained(
        model_source,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    image_size = 224

    local = Path(model_source)
    if local.exists():
        preprocessor_path = local / "preprocessor_config.json"
        if preprocessor_path.exists():
            with preprocessor_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            size = data.get("size", {})
            image_size = int(size.get("height") or size.get("width") or 224)
            mean = list(data.get("image_mean", mean))
            std = list(data.get("image_std", std))
    else:
        # For a Hugging Face-hosted checkpoint, use conventional ViT defaults
        # unless the checkpoint config exposes matching values.
        cfg = getattr(model, "config", None)
        if cfg is not None:
            image_size = int(getattr(cfg, "image_size", image_size) or image_size)

    transform = transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return model, device, transform, image_size

def preprocess(image: Image.Image, transform, device):
    image = ImageOps.exif_transpose(image).convert("RGB")
    return transform(image).unsqueeze(0).to(device)

@torch.no_grad()
def predict(image: Image.Image, model, device, transform):
    tensor = preprocess(image, transform, device)
    logits = model(pixel_values=tensor).logits
    p_manipulated = float(torch.softmax(logits, dim=1)[0, 1].cpu().item())
    label = int(p_manipulated >= 0.5)
    return label, p_manipulated

def attention_rollout(image: Image.Image, model, device, transform):
    tensor = preprocess(image, transform, device)
    with torch.no_grad():
        outputs = model(pixel_values=tensor, output_attentions=True)

    attentions = outputs.attentions
    if attentions is None:
        return None

    rollout = torch.eye(attentions[0].shape[-1], device=device)
    for attention in attentions:
        attn = attention[0].mean(dim=0)
        attn = attn + torch.eye(attn.shape[0], device=device)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        rollout = attn @ rollout

    cls_attention = rollout[0, 1:].detach().cpu().numpy()
    side = int(np.sqrt(cls_attention.shape[0]))
    heatmap = cls_attention[: side * side].reshape(side, side)
    heatmap -= heatmap.min()
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap.astype(np.float32)

def make_overlay(image: Image.Image, heatmap: np.ndarray | None, size: int):
    rgb = np.asarray(
        ImageOps.exif_transpose(image).convert("RGB").resize(
            (size, size), Image.Resampling.BILINEAR
        ),
        dtype=np.uint8,
    )
    if heatmap is None:
        return rgb

    heatmap = cv2.resize(
        heatmap, (size, size), interpolation=cv2.INTER_CUBIC
    )
    heatmap = np.clip(heatmap, 0.0, 1.0)
    color = cv2.applyColorMap(
        np.uint8(255 * heatmap), cv2.COLORMAP_JET
    )
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    return np.uint8(0.58 * rgb + 0.42 * color)

def analyse_video(uploaded_file, model, device, transform, image_size,
                  sample_every_sec: float, max_frames: int):
    suffix = Path(uploaded_file.name).suffix.lower() or ".mp4"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise RuntimeError("The uploaded video could not be opened.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total_frames / fps if total_frames > 0 else 0.0
        step = max(1, int(round(fps * sample_every_sec)))

        rows = []
        sampled_images = []
        frame_index = 0

        while len(rows) < max_frames:
            ok = cap.grab()
            if not ok:
                break

            if frame_index % step == 0:
                ok, bgr = cap.retrieve()
                if ok:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(rgb)
                    label, prob = predict(image, model, device, transform)
                    rows.append(
                        {
                            "frame_index": frame_index,
                            "time_sec": frame_index / fps,
                            "prediction": LABELS[label],
                            "manipulation_probability": prob,
                        }
                    )
                    sampled_images.append((image.copy(), prob, frame_index, frame_index / fps))
            frame_index += 1

        cap.release()

        if not rows:
            raise RuntimeError("No frames could be extracted from the video.")

        top = sorted(sampled_images, key=lambda x: x[1], reverse=True)[:5]
        peak = max(r["manipulation_probability"] for r in rows)
        overall_label = "Manipulated" if peak >= 0.5 else "Real"

        top_results = []
        for image, prob, idx, t in top:
            heatmap = attention_rollout(image, model, device, transform)
            overlay = make_overlay(image, heatmap, image_size)
            top_results.append((overlay, prob, idx, t))

        return {
            "label": overall_label,
            "probability": peak,
            "duration": duration,
            "rows": rows,
            "top_results": top_results,
            "suspicious_count": sum(
                r["manipulation_probability"] >= 0.5 for r in rows
            ),
        }
    finally:
        # Uploaded video is used only as a temporary file and removed immediately.
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

def result_banner(label: str, probability: float):
    if label == "Manipulated":
        st.error(f"Prediction: **Manipulated** — manipulation score {probability:.1%}")
    else:
        st.success(f"Prediction: **Real** — manipulation score {probability:.1%}")

st.title("Media Manipulation Forensics")
st.caption("ViT-based research prototype for image and video screening")

st.info(
    "Research demonstration only. A model prediction is not a definitive forensic "
    "determination of authenticity. The application code does not intentionally "
    "retain uploaded media after analysis."
)

with st.expander("Privacy and interpretation"):
    st.write(
        "When this tool is hosted online, the selected file is transmitted to the "
        "hosting server for inference. Images are processed in memory. Videos are "
        "written only to a temporary server-side file required for frame extraction "
        "and deleted immediately after analysis. No upload history or permanent "
        "results directory is created by this application."
    )
    st.write(
        "Attention visualisations show regions that influenced the model and should "
        "not be interpreted as proof that those pixels contain manipulation artefacts."
    )

model_source = get_model_source()
try:
    model, device, transform, image_size = load_detector(model_source)
except Exception as exc:
    st.error(
        "The ViT checkpoint could not be loaded. Configure MODEL_SOURCE in "
        "Streamlit Secrets (or FORENSIC_MODEL_SOURCE locally) to point to your "
        f"saved ViT checkpoint.\n\nDetails: {exc}"
    )
    st.stop()

st.sidebar.header("Model")
st.sidebar.write(f"Device: `{device}`")
st.sidebar.write(f"Checkpoint: `{model_source}`")

uploaded = st.file_uploader(
    "Upload an image or video",
    type=IMAGE_TYPES + VIDEO_TYPES,
    accept_multiple_files=False,
)

if uploaded is not None:
    suffix = Path(uploaded.name).suffix.lower()

    if suffix in {f".{x}" for x in IMAGE_TYPES}:
        try:
            image = Image.open(io.BytesIO(uploaded.getvalue()))
            label_id, probability = predict(image, model, device, transform)
            label = LABELS[label_id]
            heatmap = attention_rollout(image, model, device, transform)
            overlay = make_overlay(image, heatmap, image_size)

            result_banner(label, probability)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Uploaded image")
                st.image(ImageOps.exif_transpose(image).convert("RGB"), use_container_width=True)
            with col2:
                st.subheader("Attention rollout")
                st.image(overlay, use_container_width=True)

            st.caption(
                "The attention map is an exploratory explanation of model behaviour, "
                "not a localisation ground truth."
            )
        except Exception as exc:
            st.error(f"Image analysis failed: {exc}")

    elif suffix in {f".{x}" for x in VIDEO_TYPES}:
        c1, c2 = st.columns(2)
        with c1:
            sample_every_sec = st.number_input(
                "Sampling interval (seconds)", min_value=0.25, max_value=10.0,
                value=1.0, step=0.25
            )
        with c2:
            max_frames = st.number_input(
                "Maximum sampled frames", min_value=4, max_value=64,
                value=32, step=4
            )

        if st.button("Analyse video", type="primary"):
            try:
                with st.spinner("Analysing sampled frames..."):
                    result = analyse_video(
                        uploaded, model, device, transform, image_size,
                        float(sample_every_sec), int(max_frames)
                    )

                result_banner(result["label"], result["probability"])
                st.write(
                    f"Sampled frames: **{len(result['rows'])}** · "
                    f"Suspicious sampled frames: **{result['suspicious_count']}**"
                )

                st.subheader("Most suspicious sampled frames")
                cols = st.columns(min(5, len(result["top_results"])))
                for col, (overlay, prob, idx, t) in zip(cols, result["top_results"]):
                    with col:
                        st.image(overlay, use_container_width=True)
                        st.caption(f"{t:.2f}s · frame {idx} · {prob:.1%}")

                st.subheader("Per-frame results")
                st.dataframe(result["rows"], use_container_width=True)

                csv_lines = [
                    "frame_index,time_sec,prediction,manipulation_probability"
                ]
                for row in result["rows"]:
                    csv_lines.append(
                        f"{row['frame_index']},{row['time_sec']:.3f},"
                        f"{row['prediction']},{row['manipulation_probability']:.8f}"
                    )
                st.download_button(
                    "Download frame report (CSV)",
                    data="\n".join(csv_lines),
                    file_name="frame_report.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.error(f"Video analysis failed: {exc}")
