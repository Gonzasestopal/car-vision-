import io
import os
import tempfile

import numpy as np
import streamlit as st
# Optional: timm + torchvision for classifier
import timm
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from ultralytics import YOLO

st.set_page_config(page_title="Kar Vision — Parts + Model/Year", page_icon="🚗", layout="wide")
st.title("🚗 Kar Vision — Car Parts + Model/Year")
st.caption("Runs YOLO segmentation for parts and a classifier for make/model/year.")

# ----------------------------- Sidebar: settings -----------------------------
with st.sidebar:
    st.header("⚙️ Models & Settings")

    # Segmentation weights
    seg_mode = st.radio("Segmentation weights", ["Use ./segmentation.pt", "Upload .pt"], horizontal=True)
    seg_weights_path = "segmentation.pt"
    if seg_mode == "Upload .pt":
        seg_up = st.file_uploader("Upload YOLO segmentation .pt", type=["pt"])
        if seg_up:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
            tmp.write(seg_up.read()); tmp.flush()
            seg_weights_path = tmp.name

    # Classifier weights
    cls_mode = st.radio("Classifier weights", ["Use ./classifier.pt", "Upload .pt"], horizontal=True)
    cls_weights_path = "classifier.pt"
    if cls_mode == "Upload .pt":
        cls_up = st.file_uploader("Upload classifier .pt", type=["pt"])
        if cls_up:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
            tmp.write(cls_up.read()); tmp.flush()
            cls_weights_path = tmp.name

    # If the classifier is a checkpoint (state_dict), we need the backbone name
    backbone = st.text_input("Classifier backbone (timm)", value="efficientnet_b0",
                             help="Used if your classifier .pt is a checkpoint state_dict (not TorchScript).")
    img_size = st.number_input("Classifier input size", min_value=160, max_value=640, value=224, step=8)

    topk = st.slider("Top-K classes to display", 1, 10, 5)
    conf = st.slider("YOLO confidence", 0.05, 0.95, 0.35, 0.05)
    iou  = st.slider("YOLO IoU (NMS)", 0.10, 0.90, 0.50, 0.05)

# ------------------------------ Device & cache -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"**Device:** `{device}`")

@st.cache_resource(show_spinner=True)
def load_yolo(weights_path: str):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Segmentation weights not found: {weights_path}")
    m = YOLO(weights_path)  # yolo -seg weights supported
    m.fuse()
    return m

@st.cache_resource(show_spinner=True)
def load_classifier(weights_path: str, backbone: str, num_classes_hint: int | None = None):
    """
    Loads classifier in 3 ways (tries in order):
      1) TorchScript (torch.jit.load)
      2) Checkpoint dict with keys {'model': state_dict, 'classes': [...]}
      3) A full saved model object (torch.save(model))
    Returns: (callable_model, classes or None, is_torchscript: bool)
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Classifier weights not found: {weights_path}")

    # 1) TorchScript
    try:
        ts_model = torch.jit.load(weights_path, map_location=device)
        ts_model.eval()
        return ts_model, None, True
    except Exception:
        pass

    # 2) Checkpoint dict
    ckpt = torch.load(weights_path, map_location="cpu")
    if isinstance(ckpt, dict) and ("model" in ckpt):
        classes = ckpt.get("classes", None)
        if classes is None and num_classes_hint is None:
            st.warning("No 'classes' found in checkpoint. Top-K will show class indices.")
        n_classes = len(classes) if classes is not None else (num_classes_hint or 1000)
        model = timm.create_model(backbone, pretrained=False, num_classes=n_classes)
        model.load_state_dict(ckpt["model"], strict=False)
        model.to(device).eval()
        return model, classes, False

    # 3) Full model object
    if hasattr(ckpt, "forward"):
        model = ckpt.to(device).eval()
        classes = getattr(model, "classes", None)
        return model, classes, False

    raise RuntimeError("Unsupported classifier file format. "
                       "Use TorchScript (.pt via torch.jit.save) or a checkpoint with {'model','classes'}.")

# Load models
try:
    seg_model = load_yolo(seg_weights_path)
    st.success(f"Loaded segmentation weights: `{seg_weights_path}`")
except Exception as e:
    st.error("Failed to load segmentation model.")
    st.exception(e)
    st.stop()

try:
    cls_model, cls_classes, cls_is_ts = load_classifier(cls_weights_path, backbone)
    st.success(f"Loaded classifier weights: `{cls_weights_path}`")
except Exception as e:
    st.error("Failed to load classifier.")
    st.exception(e)
    st.stop()

# ---------------------------- Classifier transforms --------------------------
infer_tfms = transforms.Compose([
    transforms.Resize(int(img_size * 1.15)),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

@torch.no_grad()
def classify_pil(img: Image.Image, topk: int = 5):
    x = infer_tfms(img.convert("RGB")).unsqueeze(0).to(device)
    logits = cls_model(x)
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    probs = torch.softmax(logits, dim=1).squeeze(0)
    vals, idx = probs.topk(topk)
    idx = idx.cpu().tolist()
    vals = vals.cpu().tolist()
    if cls_classes is None:
        names = [f"class_{i}" for i in idx]
    else:
        # guard against out-of-range indices
        names = [str(cls_classes[i]) if i < len(cls_classes) else f"class_{i}" for i in idx]
    return list(zip(names, [float(v) for v in vals]))

# ------------------------------- Upload images -------------------------------
uploads = st.file_uploader(
    "Upload one or more car images",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

# ------------------------------- Inference loop ------------------------------
if uploads:
    for f in uploads:
        try:
            img = Image.open(f).convert("RGB")
        except UnidentifiedImageError:
            st.warning(f"Skipping unsupported image: {getattr(f, 'name', 'file')}")
            continue

        arr = np.array(img)

        # --- Segmentation (YOLO) ---
        try:
            seg_res = seg_model.predict(
                arr, conf=conf, iou=iou,
                device=0 if device == "cuda" else None,
                verbose=False
            )[0]
            seg_vis = seg_res.plot()[:, :, ::-1]  # BGR -> RGB
        except Exception as e:
            st.warning(f"Segmentation failed on {getattr(f, 'name', 'image')}: {e}")
            seg_vis = arr

        # --- Classification (model/year) ---
        try:
            topk_preds = classify_pil(img, topk=topk)
        except Exception as e:
            topk_preds = []
            st.warning(f"Classification failed on {getattr(f, 'name', 'image')}: {e}")

        # --- Display ---
        left, right = st.columns(2)
        with left:
            st.image(img, caption=f"Original — {getattr(f, 'name', 'image')}", use_container_width=True)
            if topk_preds:
                st.markdown("**Model/Year — Top-K predictions**")
                for name, p in topk_preds:
                    st.write(f"- {name} — `{p:.3f}`")
        with right:
            st.image(seg_vis, caption="Parts segmentation (YOLO)", use_container_width=True)

        # Optional: list parts (classes) found by YOLO
        names = seg_res.names if 'seg_res' in locals() else {}
        found = []
        if 'seg_res' in locals() and seg_res.boxes is not None and len(seg_res.boxes) > 0:
            for b in seg_res.boxes:
                cls_id = int(b.cls[0].item())
                cls_name = names.get(cls_id, str(cls_id))
                found.append(cls_name)
        if found:
            st.write("**Parts detected:**")
            for cls_name in sorted(set(found)):
                st.write(f"- {cls_name}")
else:
    st.info("Upload images to see parts segmentation and model/year predictions.")
