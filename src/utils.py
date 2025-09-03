import numpy as np


def unflatten_images(images: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """
    unflatten a batch of flattened images (B, T, C) into shape (B, H, W, C)
    """
    if len(images.shape) == 2:
        images = np.expand_dims(images, axis=-1)
    full_shape = (images.shape[0], *shape, images.shape[-1])
    return images.reshape(*full_shape)


def bw_to_rgb(images: np.ndarray) -> np.ndarray:
    """
    convert a batch of black and white images (B, H, W) to RGB (B, H, W, 3)
    """
    return np.repeat(images, 3, axis=-1)
