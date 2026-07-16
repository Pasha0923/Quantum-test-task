from pathlib import Path
import numpy as np
import rasterio

def find_band(img_dir: Path, band: str) -> Path:
    """
    Find Sentinel-2 band inside IMG_DATA directory.
    Returns Path to requested band.
    """

    files = sorted(img_dir.glob(f"*_{band}.jp2"))

    if not files:
        raise FileNotFoundError(
            f"Band {band} not found."
        )

    return files[0]

def stretch(
    channel: np.ndarray,
    lower_percentile: float = 2,
    upper_percentile: float = 98,
) -> np.ndarray:
    """
    Contrast stretching using percentiles.
    Returns values in range [0, 1].
    """

    channel = channel.astype(np.float32)

    p_low = np.percentile(channel, lower_percentile)
    p_high = np.percentile(channel, upper_percentile)

    channel = np.clip(channel, p_low, p_high)

    channel = (channel - p_low) / (p_high - p_low + 1e-8)

    return channel


def read_rgb_bands(img_dir: Path) -> np.ndarray:
    """
    Read RGB image from Sentinel-2 B04/B03/B02 bands.

    Returns  np.ndarray
        RGB image in range [0, 1].  
    """

    red_path = find_band(img_dir, "B04")
    green_path = find_band(img_dir, "B03")
    blue_path = find_band(img_dir, "B02")

    with rasterio.open(red_path) as src:
        red = src.read(1)

    with rasterio.open(green_path) as src:
        green = src.read(1)

    with rasterio.open(blue_path) as src:
        blue = src.read(1)

    rgb = np.dstack(
        (
            stretch(red),
            stretch(green),
            stretch(blue),
        )
    )

    return rgb

def find_img_data(safe_dir: Path) -> Path:
    """
    Find IMG_DATA directory inside Sentinel-2 .SAFE folder.
    """

    if not safe_dir.exists():
        raise FileNotFoundError(
            f"SAFE directory not found:\n{safe_dir}"
        )

    img_data_dirs = list(safe_dir.rglob("IMG_DATA"))

    if len(img_data_dirs) == 0:
        raise FileNotFoundError(
            f"IMG_DATA directory not found inside:\n{safe_dir}"
        )

    if len(img_data_dirs) > 1:
        raise RuntimeError(
            "Multiple IMG_DATA directories found."
        )

    return img_data_dirs[0]
