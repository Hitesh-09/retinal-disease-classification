from src.config import CLASS_NAMES, DEFAULT_MODEL_KEYS, MODELS_DIR
from src.losses import weighted_binary_crossentropy
from src.models import build_model
from src.utils import ensure_directory, resolve_model_path


def main():
    ensure_directory(MODELS_DIR)

    for model_key in DEFAULT_MODEL_KEYS:
        model = build_model(
            model_key=model_key,
            input_shape=(224, 224, 3),
            num_classes=len(CLASS_NAMES),
        )
        model.compile(
            optimizer="adam",
            loss=weighted_binary_crossentropy(),
            metrics=["AUC"],
        )
        output_path = resolve_model_path(model_key, MODELS_DIR)
        model.save(output_path)
        print(f"Saved {model_key} scaffold to {output_path}")


if __name__ == "__main__":
    main()
