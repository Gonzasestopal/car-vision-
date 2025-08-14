import numpy as np
import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Kopiloto Vision — Car Parts", page_icon="🚗", layout="wide")
st.title("🚗 Kopiloto Vision — Car Parts Segmentation")
st.caption("Highlights car parts using a YOLO segmentation model trained on Ultralytics Carparts-Seg.")

@st.cache_resource
def load_model(weights: str):
    m = YOLO(weights)
    m.fuse()
    return m

weights = st.text_input("Model weights path", value="model.pt")
model = load_model(weights)

conf = st.slider("Confidence", 0.05, 0.95, 0.35, 0.05)
iou  = st.slider("IoU (NMS)", 0.10, 0.90, 0.50, 0.05)
device = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"**Device:** `{device}`")

uploads = st.file_uploader("Upload one or more car images", type=["jpg","jpeg","png"], accept_multiple_files=True)

if uploads:
    for f in uploads:
        img = Image.open(f).convert("RGB")
        arr = np.array(img)

        # Use segmentation mode weights (yolo11*-seg.pt fine-tuned)
        res = model.predict(arr, conf=conf, iou=iou, device=0 if device=="cuda" else None, verbose=False)[0]

        # res.plot() returns an image with masks/boxes overlaid (BGR). Convert to RGB.
        plotted = res.plot()[:, :, ::-1]

        left, right = st.columns(2)
        with left:
            st.image(img, caption=f"Original — {f.name}", use_container_width=True)
        with right:
            st.image(plotted, caption="Detected parts (segmentation masks)", use_container_width=True)

        # Optional: list classes found
        names = res.names
        if res.boxes is not None and len(res.boxes) > 0:
            st.write("**Parts detected:**")
            for b in res.boxes:
                cls_id = int(b.cls[0].item())
                cls_name = names.get(cls_id, str(cls_id))
                st.write(f"- {cls_name}")
else:
    st.info("Upload images to see segmented parts.")
