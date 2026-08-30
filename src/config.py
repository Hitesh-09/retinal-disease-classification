from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
APP_DIR = PROJECT_ROOT / "app"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_IMAGE_DIR = RAW_DATA_DIR / "images"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
TESTS_DIR = PROJECT_ROOT / "tests"

CLASS_NAMES: Tuple[str, ...] = ("N", "D", "G", "C", "A", "H", "M", "O")
CLASS_DISPLAY_NAMES: Dict[str, str] = {
    "N": "Normal",
    "D": "Diabetes",
    "G": "Glaucoma",
    "C": "Cataract",
    "A": "AMD",
    "H": "Hypertension",
    "M": "Myopia",
    "O": "Other",
}

DEFAULT_IMAGE_COLUMN = "filename"
FALLBACK_IMAGE_COLUMNS: Tuple[str, ...] = ("filename", "Image", "filepath")
DEFAULT_IMAGE_SIZE: Tuple[int, int] = (224, 224)
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 1
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_THRESHOLD = 0.5
DEFAULT_RANDOM_STATE = 42
DEFAULT_TRAIN_BASE = False

DEFAULT_TRAIN_CSV = PROCESSED_DATA_DIR / "train.csv"
DEFAULT_VAL_CSV = PROCESSED_DATA_DIR / "val.csv"
DEFAULT_TEST_CSV = PROCESSED_DATA_DIR / "test.csv"

POSITIVE_CLASS_WEIGHTS: Tuple[float, ...] = (
    1.9570,
    2.0435,
    15.9470,
    15.5092,
    19.0628,
    31.8971,
    20.6135,
    3.0673,
)


@dataclass(frozen=True)
class ModelConfig:
    key: str
    display_name: str
    filename: str
    gradcam_layer: str


MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "efficientnet_b4": ModelConfig(
        key="efficientnet_b4",
        display_name="EfficientNet-B4",
        filename="efficientnet_b4_weighted.keras",
        gradcam_layer="top_conv",
    ),
    "densenet121": ModelConfig(
        key="densenet121",
        display_name="DenseNet121",
        filename="densenet_final.keras",
        gradcam_layer="conv5_block16_concat",
    ),
}

DEFAULT_MODEL_KEYS: Tuple[str, ...] = tuple(MODEL_CONFIGS.keys())
DEFAULT_GRADCAM_MODEL_KEY = "efficientnet_b4"
