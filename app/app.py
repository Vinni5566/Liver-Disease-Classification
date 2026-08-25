"""Streamlit interface for AI-assisted liver ultrasound classification."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import CLASS_NAMES, build_transforms
from models import build_model
from utils.gradcam import GradCAM, overlay_heatmap

CONFIDENCE_THRESHOLD = 0.70
MODEL_OPTIONS = {
    "ResNet50": "models/resnet50/staged_transfer/stage2_finetune/best_model.pt",
    "EfficientNet-B0": "models/efficientnet_b0/staged_transfer/stage2_finetune/best_model.pt",
}

st.set_page_config(page_title="AI-Assisted Liver Disease Classification", page_icon="medical", layout="wide")


@st.cache_resource
def load_checkpoint(model_label: str):
    """Load one of the fixed final Stage-2 checkpoints lazily."""
    path = PROJECT_ROOT / MODEL_OPTIONS[model_label]
    if not path.exists():
        return None, "The selected final model checkpoint is unavailable. Please contact the project administrator."
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        if checkpoint.get("class_names") != CLASS_NAMES:
            raise ValueError("The checkpoint class order is incompatible with this application.")
        model = build_model(config["model_name"], freeze_backbone=False).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
    except (KeyError, RuntimeError, ValueError, OSError) as exc:
        return None, f"The selected model checkpoint could not be loaded safely: {exc}"
    return (model, device, config), None


st.title("AI-Assisted Liver Disease Classification from Ultrasound Images")
st.caption("Academic research prototype — not a clinical diagnostic device.")
model_label = st.selectbox("Select model", list(MODEL_OPTIONS))
threshold = st.slider("Manual-review confidence threshold", 0.50, 0.95, CONFIDENCE_THRESHOLD, 0.05)
uploaded = st.file_uploader("Upload a single ultrasound image", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload a JPG or PNG ultrasound image to start.")
    st.stop()

loaded, error = load_checkpoint(model_label)
if error:
    st.error(error)
    st.stop()
model, device, checkpoint_config = loaded

try:
    original = Image.open(uploaded).convert("RGB")
except Exception:
    st.error("The uploaded file could not be opened as a valid image. Please upload a JPG or PNG ultrasound image.")
    st.stop()

# Matches evaluation preprocessing: resize, ToTensor, and ImageNet normalization only.
tensor = build_transforms(checkpoint_config.get("image_size", 224), training=False)(original).unsqueeze(0).to(device)
with st.spinner("Classifying image and generating Grad-CAM..."):
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    predicted_index = int(probabilities.argmax())
    confidence = float(probabilities[predicted_index])
    heatmap, _, _ = GradCAM(model)(tensor, predicted_index)

original_array = np.asarray(original)
overlay = overlay_heatmap(original_array, heatmap)
left, right = st.columns(2)
with left:
    st.subheader("Original ultrasound")
    st.image(original_array, use_container_width=True)
with right:
    st.subheader("Grad-CAM attribution")
    st.image(overlay, use_container_width=True)

st.subheader("Prediction")
metric_a, metric_b = st.columns(2)
metric_a.metric("Predicted class", CLASS_NAMES[predicted_index])
metric_b.metric("Confidence", f"{confidence * 100:.1f}%")
if confidence < threshold:
    st.warning("Low confidence — recommend manual review")
else:
    st.success("Confidence is above the configured manual-review threshold.")

probability_columns = st.columns(3)
for column, class_name, probability in zip(probability_columns, CLASS_NAMES, probabilities):
    column.metric(f"{class_name} probability", f"{probability * 100:.1f}%")
st.bar_chart({name: float(probability) for name, probability in zip(CLASS_NAMES, probabilities)})
st.caption(
    "Grad-CAM is a model-attribution visualization: it highlights regions that influenced "
    "this prediction, but does not establish medical causality."
)
