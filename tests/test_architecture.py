from src.config import CLASS_NAMES, DEFAULT_MODEL_KEYS, MODEL_CONFIGS
from src.utils import resolve_model_path


def test_config_is_single_source_of_truth():
    assert len(CLASS_NAMES) == 8
    assert tuple(MODEL_CONFIGS.keys()) == DEFAULT_MODEL_KEYS


def test_model_paths_follow_config_filenames():
    assert resolve_model_path("efficientnet_b4").name == "efficientnet_b4_weighted.keras"
    assert resolve_model_path("densenet121").name == "densenet_final.keras"
