from typing import Iterable

import tensorflow as tf

from .config import POSITIVE_CLASS_WEIGHTS


def weighted_binary_crossentropy(
    pos_weights: Iterable[float] = POSITIVE_CLASS_WEIGHTS,
):
    pos_weights_tensor = tf.constant(list(pos_weights), dtype=tf.float32)

    def loss(y_true, y_pred):
        y_true_cast = tf.cast(y_true, tf.float32)
        y_pred_clipped = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        weighted_terms = -(
            pos_weights_tensor * y_true_cast * tf.math.log(y_pred_clipped)
            + (1.0 - y_true_cast) * tf.math.log(1.0 - y_pred_clipped)
        )
        return tf.reduce_mean(weighted_terms)

    loss.__name__ = "weighted_binary_crossentropy"
    return loss


def binary_focal_loss(alpha: float = 0.25, gamma: float = 2.0):
    def loss(y_true, y_pred):
        y_true_cast = tf.cast(y_true, tf.float32)
        y_pred_clipped = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -(
            y_true_cast * tf.math.log(y_pred_clipped)
            + (1.0 - y_true_cast) * tf.math.log(1.0 - y_pred_clipped)
        )
        p_t = y_true_cast * y_pred_clipped + (1.0 - y_true_cast) * (1.0 - y_pred_clipped)
        alpha_factor = y_true_cast * alpha + (1.0 - y_true_cast) * (1.0 - alpha)
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        return tf.reduce_mean(alpha_factor * modulating_factor * cross_entropy)

    loss.__name__ = "binary_focal_loss"
    return loss
