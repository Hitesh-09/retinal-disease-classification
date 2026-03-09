from typing import List, Tuple

import tensorflow as tf

from .preprocess import CLASS_NAMES, get_paths_and_labels


def decode_image(image_path: tf.Tensor, img_size: Tuple[int, int]) -> tf.Tensor:
    """
    Decode a JPEG image, resize to img_size, and normalize to [0, 1].
    """
    image_bytes = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)  # scales to [0, 1]
    image = tf.image.resize(image, img_size)
    return image


def augment_image(image: tf.Tensor) -> tf.Tensor:
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image


def build_tf_dataset(
    csv_path: str,
    image_root: str,
    image_column: str = "Image",
    class_names: List[str] = None,
    img_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    shuffle: bool = True,
    augment: bool = False,
) -> Tuple[tf.data.Dataset, int]:
    """
    Build a tf.data pipeline that:
    - reads image paths and multi-label targets from an ODIR CSV,
    - resizes images to img_size (default 224x224),
    - normalizes pixel values to [0, 1],
    - optionally applies data augmentation.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    image_paths, labels = get_paths_and_labels(
        csv_path=csv_path,
        image_root=image_root,
        image_column=image_column,
        class_names=class_names,
    )

    num_samples = len(image_paths)

    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((path_ds, label_ds))

    def _load(path, label):
        image = decode_image(path, img_size=img_size)
        if augment:
            image_aug = augment_image(image)
        else:
            image_aug = image
        return image_aug, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=min(1000, num_samples))

    ds = ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds, num_samples


__all__ = ["build_tf_dataset", "augment_image", "decode_image"]