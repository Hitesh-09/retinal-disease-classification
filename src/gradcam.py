import cv2
import numpy as np
import tensorflow as tf

from .config import MODEL_CONFIGS


def generate_gradcam_heatmap(
    model: tf.keras.Model,
    image_batch: np.ndarray,
    layer_name: str,
) -> np.ndarray:
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(image_batch)
        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    gradients = tape.gradient(loss, conv_output)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_gradients[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
    return heatmap.numpy()


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = colored_heatmap * alpha + image
    return np.clip(overlay, 0, 255).astype("uint8")


def gradcam_for_model_key(
    model: tf.keras.Model,
    image_batch: np.ndarray,
    model_key: str,
) -> np.ndarray:
    layer_name = MODEL_CONFIGS[model_key].gradcam_layer
    return generate_gradcam_heatmap(model, image_batch=image_batch, layer_name=layer_name)
