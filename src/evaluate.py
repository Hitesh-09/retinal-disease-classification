import argparse
import os
from typing import Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

from .dataset import build_tf_dataset
from .model import binary_focal_loss
from .preprocess import CLASS_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained EfficientNet model on ODIR."
    )
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--image_column", type=str, default="Image")
    parser.add_argument("--img_size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()

    img_size: Tuple[int, int] = (args.img_size[0], args.img_size[1])

    print(f"Loading model from {args.model_path}")
   model = tf.keras.models.load_model(
    args.model_path,
    safe_mode=False
)
        custom_objects={"loss": binary_focal_loss(), "binary_focal_loss": binary_focal_loss()},
    )

    test_ds, test_count = build_tf_dataset(
        csv_path=args.test_csv,
        image_root=args.image_root,
        image_column=args.image_column,
        img_size=img_size,
        batch_size=args.batch_size,
        shuffle=False,
        augment=False,
    )

    print("Evaluating on test set...")
    results = model.evaluate(test_ds, return_dict=True)
    print("Keras evaluate metrics:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    print("Computing classification report...")
    y_true_all = []
    y_prob_all = []
    for batch_images, batch_labels in test_ds:
        probs = model.predict(batch_images, verbose=0)
        y_true_all.append(batch_labels.numpy())
        y_prob_all.append(probs)

    y_true = np.concatenate(y_true_all, axis=0)
    y_prob = np.concatenate(y_prob_all, axis=0)
    y_pred = (y_prob >= args.threshold).astype(int)

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    print("Sklearn multi-label classification report:")
    print(report)


if __name__ == "__main__":
    main()
