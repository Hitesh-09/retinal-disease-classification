import os
from typing import Tuple

import numpy as np
import streamlit as st
import tensorflow as tf

from src.model import binary_focal_loss
from src.preprocess import CLASS_NAMES


@st.cache_resource
def load_model(model_path: str) -> tf.keras.Model:
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"loss": binary_focal_loss(), "binary_focal_loss": binary_focal_loss()},
    )
    return model


def preprocess_image(image: np.ndarray, img_size: Tuple[int, int]) -> np.ndarray:
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    image = tf.image.resize(image, img_size)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image.numpy()


def main():
    st.title("Retinal Disease Classification (ODIR, EfficientNetB4)")

    model_path = st.text_input("Model path", "models/odir_efficientnet_b4_final.keras")
    img_size_str = st.text_input("Image size (H,W)", "224,224")
    h, w = [int(x.strip()) for x in img_size_str.split(",")]
    img_size = (h, w)

    uploaded_file = st.file_uploader("Upload a retinal fundus image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded image", use_column_width=True)

        if st.button("Run inference"):
            if not os.path.exists(model_path):
                st.error(f"Model not found at {model_path}")
                return

            model = load_model(model_path)

            file_bytes = uploaded_file.read()
            image = tf.image.decode_image(file_bytes, channels=3)
            image = preprocess_image(image, img_size)
            batch = np.expand_dims(image, axis=0)

            probs = model.predict(batch, verbose=0)[0]

            st.subheader("Predicted probabilities")
            for cls, p in zip(CLASS_NAMES, probs):
                st.write(f"{cls}: {p:.3f}")


if __name__ == "__main__":
    main()