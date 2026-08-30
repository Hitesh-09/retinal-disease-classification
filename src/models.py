from typing import Callable, Dict, Tuple

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.densenet import (
    DenseNet121,
    preprocess_input as densenet_preprocess_input,
)
from tensorflow.keras.applications.efficientnet import (
    EfficientNetB4,
    preprocess_input as efficientnet_preprocess_input,
)

from .config import CLASS_NAMES


def build_efficientnet_b4_classifier(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = len(CLASS_NAMES),
    train_base: bool = False,
    dropout_rate: float = 0.4,
) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape)
    x = layers.Lambda(lambda image: image * 255.0)(inputs)
    x = layers.Lambda(efficientnet_preprocess_input)(x)

    base_model = EfficientNetB4(
        include_top=False,
        input_tensor=x,
        weights="imagenet",
    )
    base_model.trainable = train_base

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="sigmoid", name="multi_label_head")(x)
    return models.Model(inputs=inputs, outputs=outputs, name="odir_efficientnet_b4")


def build_densenet121_classifier(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = len(CLASS_NAMES),
    train_base: bool = False,
    dropout_rate: float = 0.4,
) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape)
    x = layers.Lambda(lambda image: image * 255.0)(inputs)
    x = layers.Lambda(densenet_preprocess_input)(x)

    base_model = DenseNet121(
        include_top=False,
        input_tensor=x,
        weights="imagenet",
    )
    base_model.trainable = train_base

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="sigmoid", name="multi_label_head")(x)
    return models.Model(inputs=inputs, outputs=outputs, name="odir_densenet121")


MODEL_BUILDERS: Dict[str, Callable[..., tf.keras.Model]] = {
    "efficientnet_b4": build_efficientnet_b4_classifier,
    "densenet121": build_densenet121_classifier,
}


def build_model(
    model_key: str,
    input_shape: Tuple[int, int, int],
    num_classes: int = len(CLASS_NAMES),
    train_base: bool = False,
    dropout_rate: float = 0.4,
) -> tf.keras.Model:
    if model_key not in MODEL_BUILDERS:
        raise KeyError(f"Unsupported model key: {model_key}")

    return MODEL_BUILDERS[model_key](
        input_shape=input_shape,
        num_classes=num_classes,
        train_base=train_base,
        dropout_rate=dropout_rate,
    )
