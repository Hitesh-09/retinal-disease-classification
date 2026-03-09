from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.efficientnet import (
    EfficientNetB4,
    preprocess_input,
)

from .preprocess import CLASS_NAMES


def build_efficientnet_b4_classifier(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = None,
    train_base: bool = False,
    dropout_rate: float = 0.4,
) -> tf.keras.Model:
    """
    EfficientNetB4-based multi-label classifier with transfer learning,
    global average pooling, and sigmoid outputs.
    """
    if num_classes is None:
        num_classes = len(CLASS_NAMES)

    inputs = layers.Input(shape=input_shape)

    # Model expects float32 images in [0, 1]; convert to [0, 255] then apply
    # EfficientNet-specific preprocessing.
    x = layers.Lambda(lambda im: im * 255.0)(inputs)
    x = layers.Lambda(preprocess_input)(x)

    base_model = EfficientNetB4(
        include_top=False,
        input_tensor=x,
        weights="imagenet",  # transfer learning from ImageNet
    )
    base_model.trainable = train_base

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="sigmoid", name="multi_label_head")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="odir_efficientnet_b4")
    return model


def binary_focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        cross_entropy = -(
            y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        )

        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_factor = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        modulating_factor = tf.pow(1.0 - p_t, gamma)

        loss_val = alpha_factor * modulating_factor * cross_entropy
        return tf.reduce_mean(loss_val)

    return loss


__all__ = ["build_efficientnet_b4_classifier", "binary_focal_loss"]