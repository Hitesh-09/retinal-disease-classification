from pathlib import Path

import streamlit as st
from PIL import Image

from src.config import (
    CLASS_DISPLAY_NAMES,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MODEL_KEYS,
    DEFAULT_THRESHOLD,
    MODELS_DIR,
)
from src.gradcam import gradcam_for_model_key, overlay_heatmap
from src.inference import (
    default_gradcam_model_key,
    describe_model_keys,
    load_models,
    predict_image,
)


st.set_page_config(page_title="Retinal Disease Detection", layout="wide")


@st.cache_resource
def get_models(model_dir: str, model_keys: tuple[str, ...]):
    return load_models(model_keys=model_keys, model_dir=model_dir, compile=False)


def main():
    st.title("Retinal Disease Detection")
    st.caption("Training, evaluation, and UI now share the same inference utilities.")

    model_dir = st.sidebar.text_input("Model directory", str(MODELS_DIR))
    threshold = st.sidebar.slider("Prediction threshold", 0.1, 0.9, DEFAULT_THRESHOLD, 0.05)
    model_keys = tuple(DEFAULT_MODEL_KEYS)

    st.sidebar.write(f"Active models: {describe_model_keys(model_keys)}")

    try:
        models = get_models(model_dir, model_keys)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    uploaded_file = st.file_uploader("Upload a retinal image", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        st.info("Upload an image to run ensemble inference and Grad-CAM.")
        return

    image = Image.open(uploaded_file).convert("RGB")
    original_image, image_batch, prediction = predict_image(
        image=image,
        models=models,
        img_size=DEFAULT_IMAGE_SIZE,
        threshold=threshold,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input")
        st.image(original_image, use_container_width=True)

    gradcam_model_key = default_gradcam_model_key(model_keys)
    heatmap = gradcam_for_model_key(models[gradcam_model_key], image_batch, gradcam_model_key)
    overlay = overlay_heatmap(original_image, heatmap)

    with col2:
        st.subheader("Grad-CAM")
        st.image(overlay, use_container_width=True)

    st.subheader("Predictions")
    for class_name, display_name, probability in zip(
        prediction.class_names,
        prediction.display_names,
        prediction.probabilities,
    ):
        st.write(f"{display_name} ({class_name}): {probability:.3f}")
        st.progress(float(probability))

    detected = prediction.detected_labels or ["No disease detected"]
    st.success(", ".join(detected))

    st.markdown("---")
    st.caption(
        f"Models loaded from `{Path(model_dir)}` using shared src.inference utilities."
    )
    st.caption(
        "Classes: "
        + ", ".join(
            f"{class_name}={display_name}"
            for class_name, display_name in CLASS_DISPLAY_NAMES.items()
        )
    )


if __name__ == "__main__":
    main()
