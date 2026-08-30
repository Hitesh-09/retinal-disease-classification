import argparse
from typing import Tuple

import tensorflow as tf

from .config import (
    CLASS_NAMES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_IMAGE_COLUMN,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MODEL_KEYS,
    DEFAULT_TRAIN_BASE,
    DEFAULT_TRAIN_CSV,
    DEFAULT_VAL_CSV,
    MODELS_DIR,
    RAW_IMAGE_DIR,
)
from .dataset import build_tf_dataset
from .inference import compute_dataset_metrics, predict_dataset
from .losses import weighted_binary_crossentropy
from .models import build_model
from .utils import ensure_directory, resolve_model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train retinal disease classifiers using the shared project architecture."
    )
    parser.add_argument("--train_csv", type=str, default=str(DEFAULT_TRAIN_CSV))
    parser.add_argument("--val_csv", type=str, default=str(DEFAULT_VAL_CSV))
    parser.add_argument("--image_root", type=str, default=str(RAW_IMAGE_DIR))
    parser.add_argument("--image_column", type=str, default=DEFAULT_IMAGE_COLUMN)
    parser.add_argument("--img_size", type=int, nargs=2, default=list(DEFAULT_IMAGE_SIZE))
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--model_dir", type=str, default=str(MODELS_DIR))
    parser.add_argument("--model_keys", nargs="+", default=list(DEFAULT_MODEL_KEYS))
    parser.add_argument("--train_base", action="store_true")
    parser.set_defaults(train_base=DEFAULT_TRAIN_BASE)
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

    ensure_directory(args.model_dir)
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

    print(f"Training samples: {train_count}")
    print(f"Validation samples: {val_count}")

    for model_key in args.model_keys:
        print(f"\nTraining {model_key}...")
        model = build_model(
            model_key=model_key,
            input_shape=(img_size[0], img_size[1], 3),
            num_classes=num_classes,
            train_base=args.train_base,
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss=weighted_binary_crossentropy(),
            metrics=get_metrics(num_classes),
        )

        model_output_path = resolve_model_path(model_key, args.model_dir)
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_output_path),
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
            callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb],
        )

        model.save(model_output_path)

        labels, probabilities = predict_dataset([model], val_ds)
        metrics = compute_dataset_metrics(labels, probabilities)
        print(f"{model_key} saved to {model_output_path}")
        print(f"{model_key} validation macro AUC: {metrics['macro_auc']:.4f}")
        print(f"{model_key} validation binary accuracy: {metrics['binary_accuracy']:.4f}")


if __name__ == "__main__":
    main()
