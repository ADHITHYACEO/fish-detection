import os
import cv2
import time
import numpy as np
import tempfile
from pathlib import Path
import streamlit as st
from ultralytics import YOLO

# --------------------------
# App Config
# --------------------------
st.set_page_config(page_title="Fish Detection Platform", page_icon="🐟", layout="wide")

# --------------------------
# Sidebar Controls
# --------------------------
st.sidebar.title("⚙️ Settings")
WEIGHTS = st.sidebar.text_input(
    "YOLO Weights (.pt)",
    value=r"C:/Users/achua/PyCharmMiscProject/runs/detect/fish_counter_v1719/weights/best.pt",
    help="Path to your trained weights."
)
IMG_SIZE = st.sidebar.slider("Inference image size", 320, 1280, 640, step=32)
CONF_THRES = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.50, step=0.05)
IOU_THRES = st.sidebar.slider("NMS IoU threshold", 0.1, 0.9, 0.7, step=0.05)
MAX_FRAMES = st.sidebar.number_input(
    "Max frames to process (video)", min_value=1, max_value=200000, value=100000, step=100,
    help="Safety cap for very long videos."
)
show_labels = st.sidebar.checkbox("Show labels", value=True)
show_conf = st.sidebar.checkbox("Show confidence", value=True)

st.sidebar.markdown("---")
st.sidebar.write("**Tips:** For faster video processing, try smaller image size or limit frames.")

# --------------------------
# Model Loader (with caching)
# --------------------------
@st.cache_resource(show_spinner=True)
def load_model(weights_path: str):
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    model = YOLO(weights_path)
    try:
        model.fuse()
    except Exception:
        pass
    return model

# --------------------------
# Header
# --------------------------
st.title("🐟 Fish Detection Platform")
st.caption("Upload an image or a video and get back annotated results with per-frame fish counts.")

# --------------------------
# File Uploader
# --------------------------
uploaded = st.file_uploader(
    "Upload Image or Video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
    accept_multiple_files=False
)

# --------------------------
# Helper: annotate a single frame using ultralytics plot + custom HUD
# --------------------------
def annotate_frame(model, frame, img_size, conf_thres, iou_thres, show_labels=True, show_conf=True):
    results = model.predict(
        frame,
        imgsz=img_size,
        conf=conf_thres,
        iou=iou_thres,
        verbose=False
    )
    res = results[0]

    # Count fish (class id 0 assumed)
    fish_mask = (res.boxes.cls == 0) if res.boxes is not None else []
    fish_count = int(fish_mask.sum()) if len(fish_mask) else 0

    # Render default annotations
    annotated = res.plot(labels=show_labels, conf=show_conf)

    # Overlay a HUD
    cv2.putText(
        annotated,
        f"Fish: {fish_count}",
        (12, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated, fish_count


# --------------------------
# Image Flow
# --------------------------
if uploaded and uploaded.type.startswith("image"):
    st.subheader("Image Result")
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    try:
        model = load_model(WEIGHTS)
    except Exception as e:
        st.error(str(e))
        st.stop()

    with st.spinner("Running inference..."):
        annotated, fish_count = annotate_frame(
            model, bgr, IMG_SIZE, CONF_THRES, IOU_THRES, show_labels, show_conf
        )

    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption="Original", use_column_width=True)
    with col2:
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Annotated", use_column_width=True)

    # Show fish count clearly
    st.info(f"Detected Fish Count: **{fish_count}**")

    # Offer download
    out_png = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    cv2.imwrite(out_png.name, annotated)
    st.download_button("Download annotated image", data=open(out_png.name, "rb").read(), file_name="annotated.png")


# --------------------------
# Video Flow
# --------------------------
elif uploaded and uploaded.type.startswith("video"):
    st.subheader("Video Result")

    tmp_in = tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False)
    tmp_in.write(uploaded.read())
    tmp_in.flush()

    try:
        model = load_model(WEIGHTS)
    except Exception as e:
        st.error(str(e))
        st.stop()

    cap = cv2.VideoCapture(tmp_in.name)
    if not cap.isOpened():
        st.error("Could not open the uploaded video.")
        st.stop()

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    total_frames = min(total_frames, MAX_FRAMES)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    writer = cv2.VideoWriter(tmp_out.name, fourcc, fps, (width, height))

    progress = st.progress(0, text="Processing video…")
    preview = st.empty()
    live_count = st.sidebar.empty()   # Live frame count on sidebar
    total_detected = 0

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame_idx >= total_frames:
                break

            annotated, fish_count = annotate_frame(
                model, frame, IMG_SIZE, CONF_THRES, IOU_THRES, show_labels, show_conf
            )
            total_detected += fish_count

            writer.write(annotated)

            # Show preview every few frames
            if frame_idx % max(1, int(fps // 2)) == 0:
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                preview.image(rgb, caption=f"Frame {frame_idx+1}/{total_frames} — fish: {fish_count}", use_column_width=True)

            # Live per-frame count on sidebar
            live_count.info(f"Current Frame Fish Count: {fish_count}")

            # Update progress
            frame_idx += 1
            if total_frames:
                progress.progress(min(1.0, frame_idx / total_frames), text=f"Processing frame {frame_idx}/{total_frames}")

    finally:
        cap.release()
        writer.release()

    st.success("Video Processing Done!")

    # Show processed video
    st.video(tmp_out.name)

    # Display total fish count
    st.info(f"Total Detected Fish (Sum of per-frame counts): **{total_detected}**")

    # Offer download
    with open(tmp_out.name, "rb") as f:
        st.download_button("Download annotated video (MP4)", data=f, file_name="annotated.mp4")


# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.caption("This demo assumes class id 0 is 'fish'. If your dataset uses different ids/names, adjust counting logic.")
