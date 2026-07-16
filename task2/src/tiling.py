import numpy as np
from pathlib import Path
from src.io import read_rgb_bands
from src.io import find_img_data


def process_image(
    safe_dir: Path,
    tile_size: int = 1024,
    valid_threshold: float = 0.90,
) -> tuple[np.ndarray, dict]:
    """
    Read Sentinel-2 image and split it into valid tiles.

    Returns
    -------
    image : np.ndarray
        RGB image in range [0, 1].

    good_tiles : dict
        Dictionary
        key:(y, x)
        value: tile
    """

    img_dir = find_img_data(safe_dir)

    image = read_rgb_bands(img_dir)

    H, W = image.shape[:2]

    good_tiles = {}

    for y in range(0, H - tile_size + 1, tile_size):

        for x in range(0, W - tile_size + 1, tile_size):

            tile = image[y:y + tile_size,x:x + tile_size,]

            valid_mask = tile.sum(axis=2) > 0

            valid_ratio = valid_mask.mean()

            if valid_ratio >= valid_threshold:

                good_tiles[(y, x)] = tile

    print(f"Image size : {image.shape}")
    print(f"Good tiles : {len(good_tiles)}")

    return image, good_tiles