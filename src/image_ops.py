"""Small image operations shared by Question 1 processing and tests."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from astropy.io import fits
from scipy import ndimage


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def read_fits(path):
    # memmap=False is required for physical uint16 values encoded using BZERO.
    with fits.open(path, memmap=False) as hdus:
        for index, hdu in enumerate(hdus):
            if (
                isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU))
                and hdu.data is not None
                and hdu.data.ndim == 2
            ):
                return hdu.data.copy(), index
    raise ValueError(f"No two-dimensional science image HDU: {path}")


def statistics(raw):
    valid = raw[np.isfinite(raw)]
    if not valid.size:
        raise ValueError("Image has no finite pixels")
    levels = [0, 0.1, 1, 5, 50, 95, 99, 99.8, 99.9, 99.99, 100]
    return dict(
        shape_yx=list(raw.shape),
        dtype=str(raw.dtype),
        minimum=float(valid.min()),
        maximum=float(valid.max()),
        median=float(np.median(valid)),
        mean=float(np.mean(valid)),
        std=float(np.std(valid)),
        invalid_pixels=int(raw.size - valid.size),
        percentiles={
            str(p): float(v) for p, v in zip(levels, np.percentile(valid, levels))
        },
    )


def clipped_stats(values, floor):
    """MAD with clipped-standard-deviation fallback for integer quantization."""
    v = values[np.isfinite(values)].astype(np.float32)
    if not v.size:
        raise ValueError("Empty background cell")
    for _ in range(4):
        center = float(np.median(v))
        scale = 1.4826 * float(np.median(abs(v - center)))
        if scale < floor:
            lo, hi = np.percentile(v, [0.5, 99.5])
            central = v[(v >= lo) & (v <= hi)]
            scale = float(np.std(central))
        scale = max(scale, floor)
        kept = v[abs(v - center) <= 3 * scale]
        if not kept.size or kept.size == v.size:
            break
        v = kept
    return float(np.mean(v)), max(float(np.std(v)), floor)


def background_grid(array, cell, floor):
    """Smooth coarse estimates; no dependence on the 1024px export grid."""
    h, w = array.shape
    rows = list(range(0, h, cell))
    cols = list(range(0, w, cell))
    bg = np.zeros((len(rows), len(cols)), np.float32)
    noise = bg.copy()
    for iy, y in enumerate(rows):
        for ix, x in enumerate(cols):
            bg[iy, ix], noise[iy, ix] = clipped_stats(
                array[y : y + cell, x : x + cell], floor
            )
    cy = np.array([min(y + cell, h) / 2 + y / 2 - 0.5 for y in rows])
    cx = np.array([min(x + cell, w) / 2 + x / 2 - 0.5 for x in cols])
    return bg, noise, cy, cx


def grid_rows(grid, cy, cx, y0, y1, width):
    """Interpolate only a row block so a full background map is unnecessary."""
    yy = np.interp(np.arange(y0, y1), cy, np.arange(len(cy)))
    xx = np.interp(np.arange(width), cx, np.arange(len(cx)))
    return ndimage.map_coordinates(
        grid,
        [
            np.broadcast_to(yy[:, None], (y1 - y0, width)),
            np.broadcast_to(xx[None, :], (y1 - y0, width)),
        ],
        order=1,
        mode="nearest",
    )


def standardize(array, cell, floor):
    bg, noise, cy, cx = background_grid(array, cell, floor)
    result = np.empty(array.shape, np.float32)
    for y in range(0, array.shape[0], 256):
        end = min(y + 256, array.shape[0])
        result[y:end] = (
            array[y:end] - grid_rows(bg, cy, cx, y, end, array.shape[1])
        ) / grid_rows(noise, cy, cx, y, end, array.shape[1])
    return result, dict(
        background_min=float(bg.min()),
        background_max=float(bg.max()),
        noise_min=float(noise.min()),
        noise_max=float(noise.max()),
    )


def display(raw, limits):
    lo, hi = limits
    # One set of limits per source image: tiles and reconstruction match exactly.
    a = np.clip((raw.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
    a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
    return np.rint(np.arcsinh(5 * a) / np.arcsinh(5) * 255).astype(np.uint8)


def overlay(gray, mask):
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    for cls, color in [(1, [70, 220, 255]), (2, [255, 100, 70])]:
        chosen = mask == cls
        rgb[chosen] = (0.4 * rgb[chosen] + 0.6 * np.array(color)).astype(np.uint8)
    return rgb


def measurements(component, signal, raw):
    y, x = np.nonzero(component)
    weights = np.maximum(signal[component], 0.001)
    coords = np.column_stack((x, y))
    center = np.average(coords, axis=0, weights=weights)
    centered = coords - center
    cov = (centered.T * weights) @ centered / weights.sum()
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    minor, major = 4 * np.sqrt(np.maximum(eigenvalues, 0))
    ratio = float(major / max(minor, 1))
    perimeter = cv2.arcLength(
        max(
            cv2.findContours(
                component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )[0],
            key=cv2.contourArea,
        ),
        True,
    )
    return dict(
        area_px=int(component.sum()),
        width_px=int(x.max() - x.min() + 1),
        height_px=int(y.max() - y.min() + 1),
        major_axis_px=float(major),
        minor_axis_px=float(minor),
        elongation=ratio,
        eccentricity=float(np.sqrt(max(0, 1 - (minor / max(major, 1e-6)) ** 2))),
        orientation_deg=float(
            np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
        ),
        compactness=float(min(1, 4 * np.pi * component.sum() / max(perimeter**2, 1))),
        peak_intensity=float(raw[component].max()),
        mean_intensity=float(raw[component].mean()),
        peak_snr=float(signal[component].max()),
        mean_snr=float(signal[component].mean()),
    )


def mask_polygons(mask, offset_xy=(0, 0), size=1024):
    """Polygonize connected pixels, including one-pixel edge fragments.

    3x nearest-neighbour expansion gives nonzero-area contours even for a single
    pixel. Vertices lie inside their pixel cells, so OpenCV integer rasterization
    (also used by Ultralytics) reconstructs the source mask exactly. No arbitrary
    vertex stride/simplification is applied. Offsets are added BEFORE normalizing.
    """
    contours, _ = cv2.findContours(
        np.repeat(np.repeat(mask.astype(np.uint8), 3, 0), 3, 1),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    result = []
    for contour in contours:
        points = contour[:, 0, :].astype(np.float64) / 3 + 0.1
        points += np.array(offset_xy)
        if len(points) < 3 or cv2.contourArea(points.astype(np.float32)) <= 0:
            raise ValueError("Degenerate mask polygon")
        result.append(points / size)
    return result


def rasterize(points, size=1024):
    output = np.zeros((size, size), np.uint8)
    # Use float32 and truncation, matching the usual YOLO/OpenCV conversion.
    pixels = (np.asarray(points, dtype=np.float32) * size).astype(np.int32)
    cv2.fillPoly(output, [pixels], 1)
    return output.astype(bool)
