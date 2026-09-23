"""Image-processing helpers for Digantara Question 1.

The functions in this module are intentionally small and reusable. They handle
FITS loading, robust background/noise estimation, morphology measurements,
display scaling, and conversion of binary masks to YOLO segmentation polygons.
"""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from astropy.io import fits
from scipy import ndimage


def read_fits(path: Path) -> np.ndarray:
    """Return the first 2D science image found in a FITS file.

    FITS files can contain multiple HDUs. The assessment images are single-band
    2D arrays, so the first 2D image HDU is used. memmap=False is important for
    FITS files that encode unsigned 16-bit values using BZERO/BSCALE.
    """
    with fits.open(path, memmap=False) as hdus:
        for hdu in hdus:
            if (
                isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU))
                and hdu.data is not None
                and hdu.data.ndim == 2
            ):
                return np.asarray(hdu.data).copy()

    raise ValueError(f"No 2D image HDU found in: {path}")


def image_statistics(image: np.ndarray) -> dict[str, Any]:
    """Return basic statistics used to inspect a raw FITS image."""
    valid = image[np.isfinite(image)]
    if valid.size == 0:
        raise ValueError("Image contains no finite pixels.")

    percentiles = [0, 1, 5, 50, 95, 99, 99.9, 99.99, 100]
    values = np.percentile(valid, percentiles)

    return {
        "shape_yx": list(image.shape),
        "dtype": str(image.dtype),
        "minimum": float(valid.min()),
        "maximum": float(valid.max()),
        "mean": float(valid.mean()),
        "median": float(np.median(valid)),
        "std": float(valid.std()),
        "invalid_pixels": int(image.size - valid.size),
        "percentiles": {
            str(level): float(value)
            for level, value in zip(percentiles, values, strict=True)
        },
    }


def _robust_cell_stats(values: np.ndarray, noise_floor: float) -> tuple[float, float]:
    """Estimate background and noise for one image cell."""
    data = values[np.isfinite(values)].astype(np.float32)
    if data.size == 0:
        raise ValueError("Background cell contains no finite pixels.")

    center = float(np.median(data))
    mad = float(np.median(np.abs(data - center)))
    noise = 1.4826 * mad

    # Integer-valued backgrounds can make MAD equal to zero.
    if noise < noise_floor:
        low, high = np.percentile(data, [0.5, 99.5])
        central = data[(data >= low) & (data <= high)]
        noise = float(central.std()) if central.size else 0.0

    return center, max(noise, noise_floor)


def _background_grid(
    image: np.ndarray,
    cell_size: int,
    noise_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate coarse background/noise grids across the full image."""
    height, width = image.shape
    y_starts = list(range(0, height, cell_size))
    x_starts = list(range(0, width, cell_size))

    background = np.zeros((len(y_starts), len(x_starts)), dtype=np.float32)
    noise = np.zeros_like(background)

    for row, y0 in enumerate(y_starts):
        for column, x0 in enumerate(x_starts):
            block = image[y0 : y0 + cell_size, x0 : x0 + cell_size]
            background[row, column], noise[row, column] = _robust_cell_stats(
                block,
                noise_floor,
            )

    y_centers = np.array(
        [(y0 + min(y0 + cell_size, height) - 1) / 2 for y0 in y_starts]
    )
    x_centers = np.array(
        [(x0 + min(x0 + cell_size, width) - 1) / 2 for x0 in x_starts]
    )
    return background, noise, y_centers, x_centers


def _interpolate_grid_rows(
    grid: np.ndarray,
    y_centers: np.ndarray,
    x_centers: np.ndarray,
    y0: int,
    y1: int,
    width: int,
) -> np.ndarray:
    """Interpolate a coarse grid for one row block of the source image."""
    y_index = np.interp(
        np.arange(y0, y1),
        y_centers,
        np.arange(len(y_centers)),
    )
    x_index = np.interp(
        np.arange(width),
        x_centers,
        np.arange(len(x_centers)),
    )

    return ndimage.map_coordinates(
        grid,
        [
            np.broadcast_to(y_index[:, None], (y1 - y0, width)),
            np.broadcast_to(x_index[None, :], (y1 - y0, width)),
        ],
        order=1,
        mode="nearest",
    )


def standardize_local_background(
    image: np.ndarray,
    cell_size: int,
    noise_floor: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Convert an image to a local signal-to-noise representation.

    The returned array is approximately:
        (pixel - local_background) / local_noise
    """
    background, noise, y_centers, x_centers = _background_grid(
        image,
        cell_size,
        noise_floor,
    )

    standardized = np.empty(image.shape, dtype=np.float32)
    for y0 in range(0, image.shape[0], 256):
        y1 = min(y0 + 256, image.shape[0])

        local_background = _interpolate_grid_rows(
            background,
            y_centers,
            x_centers,
            y0,
            y1,
            image.shape[1],
        )
        local_noise = _interpolate_grid_rows(
            noise,
            y_centers,
            x_centers,
            y0,
            y1,
            image.shape[1],
        )

        standardized[y0:y1] = (
            image[y0:y1] - local_background
        ) / local_noise

    info = {
        "background_min": float(background.min()),
        "background_max": float(background.max()),
        "noise_min": float(noise.min()),
        "noise_max": float(noise.max()),
    }
    return standardized, info


def make_display_image(
    image: np.ndarray,
    lower: float,
    upper: float,
) -> np.ndarray:
    """Create an 8-bit display image without modifying scientific source values."""
    scale = max(upper - lower, 1e-6)
    normalized = np.clip(
        (image.astype(np.float32) - lower) / scale,
        0.0,
        1.0,
    )
    normalized = np.nan_to_num(
        normalized,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    stretched = np.arcsinh(5.0 * normalized) / np.arcsinh(5.0)
    return np.rint(stretched * 255).astype(np.uint8)


def overlay_mask(gray: np.ndarray, class_mask: np.ndarray) -> np.ndarray:
    """Overlay star/blob and streak masks on an 8-bit grayscale image."""
    rgb = np.repeat(gray[:, :, None], 3, axis=2)

    colors = {
        1: np.array([70, 220, 255]),
        2: np.array([255, 100, 70]),
    }
    for value, color in colors.items():
        selected = class_mask == value
        rgb[selected] = (
            0.4 * rgb[selected] + 0.6 * color
        ).astype(np.uint8)

    return rgb


def measure_component(
    component: np.ndarray,
    snr_image: np.ndarray,
    raw_image: np.ndarray,
) -> dict[str, float | int]:
    """Measure morphology and brightness features for one connected component."""
    y, x = np.nonzero(component)
    if len(x) == 0:
        raise ValueError("Cannot measure an empty component.")

    weights = np.maximum(snr_image[component], 0.001)
    coordinates = np.column_stack((x, y))
    center = np.average(coordinates, axis=0, weights=weights)
    centered = coordinates - center

    covariance = (centered.T * weights) @ centered / weights.sum()
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    minor_axis, major_axis = 4 * np.sqrt(np.maximum(eigenvalues, 0))

    contours, _ = cv2.findContours(
        component.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    perimeter = max(
        (cv2.arcLength(contour, True) for contour in contours),
        default=0.0,
    )

    elongation = float(major_axis / max(minor_axis, 1.0))
    eccentricity = float(
        np.sqrt(
            max(
                0.0,
                1.0 - (minor_axis / max(major_axis, 1e-6)) ** 2,
            )
        )
    )
    compactness = float(
        min(
            1.0,
            4.0 * np.pi * component.sum() / max(perimeter**2, 1.0),
        )
    )

    return {
        "area_px": int(component.sum()),
        "width_px": int(x.max() - x.min() + 1),
        "height_px": int(y.max() - y.min() + 1),
        "major_axis_px": float(major_axis),
        "minor_axis_px": float(minor_axis),
        "elongation": elongation,
        "eccentricity": eccentricity,
        "orientation_deg": float(
            np.degrees(
                np.arctan2(
                    eigenvectors[1, 1],
                    eigenvectors[0, 1],
                )
            )
        ),
        "compactness": compactness,
        "peak_intensity": float(raw_image[component].max()),
        "mean_intensity": float(raw_image[component].mean()),
        "peak_snr": float(snr_image[component].max()),
        "mean_snr": float(snr_image[component].mean()),
    }


def mask_to_polygons(
    mask: np.ndarray,
    offset_xy: tuple[int, int],
    tile_size: int,
) -> list[np.ndarray]:
    """Convert one binary mask fragment to normalized YOLO polygons.

    A temporary 3x expansion represents each source pixel as an area rather than
    a point. This lets very thin fragments produce valid segmentation polygons.
    """
    expanded = np.repeat(
        np.repeat(mask.astype(np.uint8), 3, axis=0),
        3,
        axis=1,
    )
    contours, _ = cv2.findContours(
        expanded,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    polygons: list[np.ndarray] = []
    for contour in contours:
        points = contour[:, 0, :].astype(np.float64) / 3.0 + 0.1
        points += np.array(offset_xy, dtype=np.float64)

        if (
            len(points) < 3
            or cv2.contourArea(points.astype(np.float32)) <= 0
        ):
            continue

        polygons.append(points / float(tile_size))

    return polygons


def rasterize_polygon(
    points: np.ndarray,
    tile_size: int,
) -> np.ndarray:
    """Rasterize one normalized segmentation polygon to a boolean mask."""
    output = np.zeros((tile_size, tile_size), dtype=np.uint8)
    pixels = (
        np.asarray(points, dtype=np.float32) * tile_size
    ).astype(np.int32)
    cv2.fillPoly(output, [pixels], 1)
    return output.astype(bool)
