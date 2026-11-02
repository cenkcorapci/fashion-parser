import numpy as np

from fashion_parser.data.torch_dataset import decode_rle
from fashion_parser.utils.image_utils import refine_masks, to_rle


def test_to_rle():
    assert to_rle([0, 1, 1, 0, 1]) == [1, 2, 4, 1]


def test_decode_rle_is_column_major_and_one_indexed():
    mask = decode_rle("1 2", height=3, width=2)
    assert mask[:, 0].tolist() == [1, 1, 0]
    assert mask[:, 1].sum() == 0


def test_refine_masks_removes_overlap():
    masks = np.zeros((4, 4, 2), dtype=bool)
    masks[:2, :2, 0] = True
    masks[:3, :3, 1] = True
    rois = np.zeros((2, 4), dtype=int)
    masks, rois = refine_masks(masks, rois)
    assert not np.any(masks[:, :, 0] & masks[:, :, 1])
    assert rois[0].tolist() == [0, 0, 1, 1]
