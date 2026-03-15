import argparse
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

from src.dataset import build_tf_dataset


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--image_column", type=str, default="filename")
    parser.add_argument("--img_size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--batch_size", type=int, default=16)

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model from {args.model_path}")

    model = tf.keras.models.load_model(
        args.model_path,
        compile=False,
        safe_mode=False,
        custom_objects={"preprocess_input": preprocess_input}
    )

    print("Building test dataset...")

    test_ds, test_count = build_tf_dataset(
        csv_path=args.test_csv,
        image_root=args.image_root,
        image_column=args.image_column,
        img_size=tuple(args.img_size),
        batch_size=args.batch_size,
        training=False
    )

    print(f"Total test samples: {test_count}")

    results = model.evaluate(test_ds)

    print("\nTest Results:")
    for name, value in zip(model.metrics_names, results):
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
