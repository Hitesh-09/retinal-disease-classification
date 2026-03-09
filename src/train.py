import argparse
import os
from typing import Tuple

import tensorflow as tf

from .dataset import build_tf_dataset
from .model import build_efficientnet_b4_classifier, binary_focal_loss
from .preprocess import CLASS_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EfficientNet multi-label classifier on ODIR."
    )
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--image_column", type=str, default="Image")
    parser.add_argument("--img_size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--train_base", action="store_true")
    return parser.parse_args()


def get_metrics(num_classes: int):
    return [
        tf.keras.metrics.AUC(
            name="auc_macro", multi_label=True, num_labels=num_classes
        ),
        tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
    ]


def main():
    args = parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    img_size: Tuple[int, int] = (args.img_size[0], args.img_size[1])
    num_classes = len(CLASS_NAMES)

    train_ds, train_count = build_tf_dataset(
        csv_path=args.train_csv,
        image_root=args.image_root,
        image_column=args.image_column,
        img_size=img_size,
        batch_size=args.batch_size,
        shuffle=True,
        augment=True,
    )

    val_ds, val_count = build_tf_dataset(
        csv_path=args.val_csv,
        image_root=args.image_root,
        image_column=args.image_column,
        img_size=img_size,
        batch_size=args.batch_size,
        shuffle=False,
        augment=False,
    )

    steps_per_epoch = max(train_count // args.batch_size, 1)
    val_steps = max(val_count // args.batch_size, 1)

    model = build_efficientnet_b4_classifier(
        input_shape=(img_size[0], img_size[1], 3),
        num_classes=num_classes,
        train_base=args.train_base,
    )

    loss_fn = binary_focal_loss(gamma=2.0, alpha=0.25)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=get_metrics(num_classes=num_classes),
    )

    checkpoint_path = os.path.join(
        args.model_dir, "odir_efficientnet_b4.{epoch:02d}-{val_auc_macro:.4f}.keras"
    )
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_auc_macro",
        mode="max",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc_macro",
        mode="max",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_auc_macro",
        mode="max",
        factor=0.5,
        patience=2,
        verbose=1,
        min_lr=1e-7,
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb],
    )

    final_model_path = os.path.join(args.model_dir, "odir_efficientnet_b4_final.keras")
    model.save(final_model_path)
    print(f"Training complete. Final model saved to {final_model_path}")


if __name__ == "__main__":
    main()