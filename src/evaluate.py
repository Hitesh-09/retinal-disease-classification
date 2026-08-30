import argparse


from src.config import (
    CLASS_DISPLAY_NAMES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_IMAGE_COLUMN,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MODEL_KEYS,
    DEFAULT_TEST_CSV,
    MODELS_DIR,
    RAW_IMAGE_DIR,
)
from src.dataset import build_tf_dataset
from src.inference import compute_dataset_metrics, describe_model_keys, load_models, predict_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one or more trained models through the shared inference stack."
    )
    parser.add_argument("--test_csv", type=str, default=str(DEFAULT_TEST_CSV))
    parser.add_argument("--image_root", type=str, default=str(RAW_IMAGE_DIR))
    parser.add_argument("--image_column", type=str, default=DEFAULT_IMAGE_COLUMN)
    parser.add_argument("--img_size", type=int, nargs=2, default=list(DEFAULT_IMAGE_SIZE))
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--model_dir", type=str, default=str(MODELS_DIR))
    parser.add_argument("--model_keys", nargs="+", default=list(DEFAULT_MODEL_KEYS))
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Loading models: {describe_model_keys(args.model_keys)}")
    models = load_models(args.model_keys, model_dir=args.model_dir, compile=False)

    test_ds, test_count = build_tf_dataset(
        csv_path=args.test_csv,
        image_root=args.image_root,
        image_column=args.image_column,
        img_size=tuple(args.img_size),
        batch_size=args.batch_size,
        shuffle=False,
        augment=False,
    )

    print(f"Total test samples: {test_count}")
    labels, probabilities = predict_dataset(models, test_ds)
    metrics = compute_dataset_metrics(labels, probabilities)

    print("\nTest Results:")
    print(f"Macro AUC: {metrics['macro_auc']:.4f}")
    print(f"Binary accuracy: {metrics['binary_accuracy']:.4f}")

    print("\nPer-class AUC:")
    for class_name, auc_value in metrics["per_class_auc"].items():
        print(f"{CLASS_DISPLAY_NAMES[class_name]} ({class_name}): {auc_value:.4f}")


if __name__ == "__main__":
    main()
