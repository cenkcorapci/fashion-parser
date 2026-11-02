import itertools

import cv2
import numpy as np

from fashion_parser.config.settings import IMAGE_SIZE


def resize_image(image_path):
    """Read an image as RGB and resize it to ``IMAGE_SIZE`` x ``IMAGE_SIZE``."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return img


def to_rle(bits):
    """Run-length encode a flat binary sequence as ``[start, length, ...]`` (0-indexed)."""
    rle = []
    pos = 0
    for bit, group in itertools.groupby(bits):
        group_list = list(group)
        if bit:
            rle.extend([pos, sum(group_list)])
        pos += len(group_list)
    return rle


def refine_masks(masks, rois):
    """Remove overlaps between ``[H, W, N]`` masks (smaller masks win) and refit ``rois``.

    The competition submission format forbids overlapping masks. ``rois`` rows are
    ``[y1, x1, y2, x2]``. Inputs are modified in place and also returned.
    """
    areas = np.sum(masks.reshape(-1, masks.shape[-1]), axis=0)
    mask_index = np.argsort(areas)
    union_mask = np.zeros(masks.shape[:-1], dtype=bool)
    for m in mask_index:
        masks[:, :, m] = np.logical_and(masks[:, :, m], np.logical_not(union_mask))
        union_mask = np.logical_or(masks[:, :, m], union_mask)
    for m in range(masks.shape[-1]):
        mask_pos = np.where(masks[:, :, m])
        if np.any(mask_pos):
            y1, x1 = np.min(mask_pos, axis=1)
            y2, x2 = np.max(mask_pos, axis=1)
            rois[m, :] = [y1, x1, y2, x2]
    return masks, rois
