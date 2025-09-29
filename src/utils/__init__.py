import imageio
import numpy as np
import torch
from imageio import mimsave

from src.utils.drawing import create_basketball_frame


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


def create_frames_from_trajectory(trajectory: np.ndarray, game: str) -> list[np.ndarray]:
    """
    create frames from a trajectory
    """
    frames = []
    for pts in trajectory:
        if game == "basketball":
            frame = create_basketball_frame(pts)
        else:
            raise ValueError(f"Unknown game: {game}")
        frames.append(frame)
    return frames


def create_video_from_frames(frames: list[np.ndarray], video_path: str, fps: int = 10):
    """
    create a video from a list of frames
    """
    mimsave(video_path, frames, fps=fps)


if __name__ == "__main__":
    trajectory = np.array(
        [
            [
                [39.2, 21.5],
                [17.9, 34.7],
                [39.3, 22.5],
                [33.4, 0.0],
                [20.7, 19.2],
                [23.0, 26.1],
                [18.9, 31.7],
                [21.0, 26.9],
                [31.9, 24.1],
                [23.3, 8.0],
                [16.0, 20.1],
            ],
            [
                [39.1, 22.1],
                [11.3, 42.8],
                [39.1, 22.3],
                [22.7, 0.1],
                [22.5, 18.2],
                [15.3, 32.9],
                [13.2, 36.6],
                [16.7, 29.2],
                [34.1, 22.5],
                [15.3, 8.9],
                [18.9, 20.9],
            ],
            [
                [36.3, 12.8],
                [9.9, 47.6],
                [34.8, 14.2],
                [11.0, 1.2],
                [30.8, 18.0],
                [5.8, 26.1],
                [11.3, 35.3],
                [8.1, 27.7],
                [33.7, 16.3],
                [8.8, 8.4],
                [29.6, 13.1],
            ],
            [
                [34.7, 10.2],
                [22.8, 46.2],
                [35.7, 9.5],
                [2.2, 1.6],
                [25.4, 23.8],
                [13.7, 14.9],
                [16.0, 30.3],
                [10.8, 21.1],
                [26.8, 8.6],
                [5.7, 7.5],
                [28.2, 7.3],
            ],
            [
                [20.9, 29.9],
                [28.7, 45.5],
                [32.5, 13.0],
                [4.9, 1.7],
                [22.0, 28.6],
                [14.5, 18.7],
                [20.9, 37.2],
                [11.9, 24.2],
                [26.1, 16.1],
                [5.5, 8.8],
                [16.4, 20.2],
            ],
        ]
    )
    frames = create_frames_from_trajectory(trajectory, "basketball")
    create_video_from_frames(frames, "/Users/wzteoh/Downloads/basketball_sample.mp4", fps=2)
