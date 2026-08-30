from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import roc_auc_score
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input

from .config import (
    CLASS_DISPLAY_NAMES,
    CLASS_NAMES,
    DEFAULT_GRADCAM_MODEL_KEY,
    DEFAULT_MODEL_KEYS,
    DEFAULT_THRESHOLD,
    MODEL_CONFIGS,
    MODELS_DIR,
)
from .losses import binary_focal_loss, weighted_binary_crossentropy
from .utils import resolve_model_path


MODEL_PREPROCESSORS = {
    "efficientnet_b4": efficientnet_preprocess_input,
    "densenet121": densenet_preprocess_input,
}


@dataclass
class PredictionResult:
    probabilities: np.ndarray
    binary: np.ndarray
    class_names: list[str]
    display_names: list[str]
    detected_labels: list[str]


def get_model_custom_objects(model_key: str) -> dict:
    if model_key not in MODEL_PREPROCESSORS:
        raise KeyError(f"Unsupported model key: {model_key}")

    return {
        "preprocess_input": MODEL_PREPROCESSORS[model_key],
        "weighted_binary_crossentropy": weighted_binary_crossentropy(),
        "binary_focal_loss": binary_focal_loss(),
    }


def load_trained_model(
    model_key: str,
    model_dir: Path | str = MODELS_DIR,
    compile: bool = False,
) -> tf.keras.Model:
    model_path = resolve_model_path(model_key, model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    return tf.keras.models.load_model(
        model_path,
        compile=compile,
        safe_mode=False,
        custom_objects=get_model_custom_objects(model_key),
    )


def load_models(
    model_keys: Sequence[str] = DEFAULT_MODEL_KEYS,
    model_dir: Path | str = MODELS_DIR,
    compile: bool = False,
) -> Dict[str, tf.keras.Model]:
    return {
        model_key: load_trained_model(model_key, model_dir=model_dir, compile=compile)
        for model_key in model_keys
    }


def prepare_image(image: Image.Image | np.ndarray | tf.Tensor, img_size: tuple[int, int]):
    if isinstance(image, Image.Image):
        image_array = np.asarray(image.convert("RGB"))
    elif tf.is_tensor(image):
        image_array = image.numpy()
    else:
        image_array = np.asarray(image)

    if image_array.ndim == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    if image_array.shape[-1] == 4:
        image_array = image_array[..., :3]

    image_array = image_array.astype("uint8")
    resized = tf.image.resize(image_array, img_size)
    normalized = tf.cast(resized, tf.float32) / 255.0
    batch = tf.expand_dims(normalized, axis=0)
    return image_array, batch.numpy()


def ensemble_predict(
    models: Mapping[str, tf.keras.Model] | Sequence[tf.keras.Model],
    batch: np.ndarray | tf.Tensor,
) -> np.ndarray:
    if isinstance(models, Mapping):
        model_list = list(models.values())
    else:
        model_list = list(models)

    predictions = [model.predict(batch, verbose=0) for model in model_list]
    return np.mean(predictions, axis=0)


def decode_predictions(
    probabilities: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> PredictionResult:
    probability_vector = np.asarray(probabilities)[0]
    binary_vector = (probability_vector >= threshold).astype("int32")
    class_names = list(CLASS_NAMES)
    display_names = [CLASS_DISPLAY_NAMES[class_name] for class_name in class_names]
    detected_labels = [
        CLASS_DISPLAY_NAMES[class_name]
        for class_name, flag in zip(class_names, binary_vector)
        if flag
    ]
    return PredictionResult(
        probabilities=probability_vector,
        binary=binary_vector,
        class_names=class_names,
        display_names=display_names,
        detected_labels=detected_labels,
    )


def predict_image(
    image: Image.Image | np.ndarray | tf.Tensor,
    models: Mapping[str, tf.keras.Model] | Sequence[tf.keras.Model],
    img_size: tuple[int, int],
    threshold: float = DEFAULT_THRESHOLD,
):
    original_image, batch = prepare_image(image, img_size=img_size)
    probabilities = ensemble_predict(models, batch)
    return original_image, batch, decode_predictions(probabilities, threshold=threshold)


def predict_dataset(
    models: Mapping[str, tf.keras.Model] | Sequence[tf.keras.Model],
    dataset: tf.data.Dataset,
) -> tuple[np.ndarray, np.ndarray]:
    labels = []
    probabilities = []

    for images, batch_labels in dataset:
        labels.append(batch_labels.numpy())
        probabilities.append(ensemble_predict(models, images))

    return np.concatenate(labels, axis=0), np.concatenate(probabilities, axis=0)


def compute_dataset_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    binary = (probabilities >= threshold).astype("int32")
    metrics = {
        "threshold": threshold,
        "binary_accuracy": float(np.mean(binary == labels)),
        "macro_auc": safe_macro_auc(labels, probabilities),
        "per_class_auc": {},
    }

    for index, class_name in enumerate(CLASS_NAMES):
        metrics["per_class_auc"][class_name] = safe_binary_auc(
            labels[:, index], probabilities[:, index]
        )

    return metrics


def safe_binary_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        return float("nan")
    return float(roc_auc_score(labels, probabilities))


def safe_macro_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    valid_scores = []
    for index in range(probabilities.shape[1]):
        score = safe_binary_auc(labels[:, index], probabilities[:, index])
        if not np.isnan(score):
            valid_scores.append(score)
    return float(np.mean(valid_scores)) if valid_scores else float("nan")


def default_gradcam_model_key(model_keys: Iterable[str]) -> str:
    keys = list(model_keys)
    if DEFAULT_GRADCAM_MODEL_KEY in keys:
        return DEFAULT_GRADCAM_MODEL_KEY
    return keys[0]


def describe_model_keys(model_keys: Sequence[str]) -> str:
    return ", ".join(MODEL_CONFIGS[model_key].display_name for model_key in model_keys)
