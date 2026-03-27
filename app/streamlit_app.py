import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Retinal Disease Detection", layout="wide")

# -----------------------------
# CUSTOM LOSS
# -----------------------------
def weighted_loss(y_true, y_pred):
    pos_weights = tf.constant([1.9570, 2.0435, 15.9470, 15.5092, 19.0628, 31.8971, 20.6135, 3.0673], dtype=tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    loss = -(pos_weights * y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
    return tf.reduce_mean(loss)

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_models():
    try:
        eff_model = tf.keras.models.load_model(
            "models/efficientnet_b4_weighted.keras",
            custom_objects={"weighted_loss": weighted_loss},
            safe_mode=False
        )

        den_model = tf.keras.models.load_model(
            "models/densenet_final.keras",
            custom_objects={"weighted_loss": weighted_loss},
            safe_mode=False
        )
        return eff_model, den_model
    except Exception as e:
        st.error(f"Models not found or error loading models: {e}")
        st.stop()

eff_model, den_model = load_models()

# -----------------------------
# CLASS LABELS
# -----------------------------
class_names = ["Normal","Diabetes","Glaucoma","Cataract",
               "AMD","Hypertension","Myopia","Other"]

# -----------------------------
# GRAD-CAM FUNCTION
# -----------------------------
def get_gradcam(img_array, model, layer_name="top_conv"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_array)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

# -----------------------------
# UI
# -----------------------------
st.title("🧠 Retinal Disease Detection System")

st.markdown("Upload a retinal image to detect possible diseases")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    # -----------------------------
    # LOAD IMAGE
    # -----------------------------
    image = Image.open(uploaded_file)
    image = np.array(image)

    img_resized = cv2.resize(image, (224,224))
    img_array = np.expand_dims(img_resized, axis=0)
    img_array = preprocess_input(img_array)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

    # -----------------------------
    # PREDICTION
    # -----------------------------
    pred_eff = eff_model.predict(img_array)
    pred_den = den_model.predict(img_array)

    pred = (pred_eff + pred_den) / 2

    threshold = 0.5
    pred_binary = (pred > threshold)[0]

    # -----------------------------
    # SHOW RESULTS
    # -----------------------------
    st.subheader("📊 Predictions")

    for i, val in enumerate(pred[0]):
        st.write(f"{class_names[i]}: {val:.2f}")
        st.progress(float(val))

    detected = [class_names[i] for i in range(len(class_names)) if pred_binary[i]]

    st.success(f"Detected: {', '.join(detected) if detected else 'No disease detected'}")

    # -----------------------------
    # GRAD-CAM
    # -----------------------------
    heatmap = get_gradcam(img_array, eff_model)

    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = heatmap * 0.4 + image

    with col2:
        st.subheader("Grad-CAM")
        st.image(superimposed.astype("uint8"))

    # -----------------------------
    # EXPLANATION
    # -----------------------------
    st.subheader("🧾 Interpretation")

    if "Diabetes" in detected:
        st.warning("Possible signs of diabetic retinopathy detected.")
    if "Glaucoma" in detected:
        st.warning("Possible glaucoma indicators found.")
    if "Normal" in detected:
        st.info("No major abnormalities detected.")

    # -----------------------------
    # MODEL INFO
    # -----------------------------
    st.markdown("---")
    st.markdown("### ℹ️ Model Info")
    st.write("Architecture: EfficientNet + DenseNet (Ensemble)")
    st.write("Test AUC: 0.865")

    # -----------------------------
    # DISCLAIMER
    # -----------------------------
    st.markdown("---")
    st.warning("⚠️ This tool is for research purposes only. Not for medical diagnosis.")


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