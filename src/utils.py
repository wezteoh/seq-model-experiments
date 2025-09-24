import numpy as np
import torch


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


def cast_floats_by_trainer_precision(item, *, precision):
    if "64" in precision:
        return cast_floats(item, device=item.device, dtype=torch.float64)
    elif "32" in precision:
        return cast_floats(item, device=item.device, dtype=torch.float32)
    elif "bf16" in precision:
        return cast_floats(item, device=item.device, dtype=torch.bfloat16)
    elif "fp16" in precision:
        return cast_floats(item, device=item.device, dtype=torch.half)
    else:
        return item


def cast_floats(item, *, device, dtype, non_blocking=True):
    if torch.is_tensor(item):
        if torch.is_floating_point(item):
            return item.to(device=device, dtype=dtype, non_blocking=non_blocking)
        else:
            return item.to(device=device, non_blocking=non_blocking)
    elif isinstance(item, dict):
        return {
            k: cast_floats(v, device=device, dtype=dtype, non_blocking=non_blocking)
            for k, v in item.items()
        }
    elif isinstance(item, (list, tuple)):
        t = [cast_floats(v, device=device, dtype=dtype, non_blocking=non_blocking) for v in item]
        return type(item)(*t) if isinstance(item, tuple) else t
    else:
        return item
