"""Unit tests for utils/image_utils.py module."""
import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock
import cv2

from utils.image_utils import resize_image, to_rle, refine_masks


class TestResizeImage:
    """Tests for resize_image function."""
    
    @patch('utils.image_utils.cv2.imread')
    @patch('utils.image_utils.cv2.cvtColor')
    @patch('utils.image_utils.cv2.resize')
    def test_resize_image_success(self, mock_resize, mock_cvtColor, mock_imread):
        """Test successful image resize."""
        # Setup
        mock_img = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = mock_img
        mock_cvtColor.return_value = mock_img
        expected_resized = np.zeros((512, 512, 3), dtype=np.uint8)
        mock_resize.return_value = expected_resized
        
        # Execute
        result = resize_image('/path/to/image.jpg')
        
        # Assert
        mock_imread.assert_called_once_with('/path/to/image.jpg')
        mock_cvtColor.assert_called_once_with(mock_img, cv2.COLOR_BGR2RGB)
        mock_resize.assert_called_once()
        assert result.shape == (512, 512, 3)
    
    @patch('utils.image_utils.cv2.imread')
    def test_resize_image_with_invalid_path(self, mock_imread):
        """Test resize_image with invalid image path."""
        mock_imread.return_value = None
        
        with pytest.raises((AttributeError, Exception)):
            resize_image('/invalid/path.jpg')


class TestToRle:
    """Tests for to_rle (run-length encoding) function."""
    
    def test_to_rle_simple_case(self):
        """Test RLE with simple binary array."""
        bits = np.array([0, 0, 1, 1, 1, 0, 1, 1])
        result = to_rle(bits)
        
        # First run of 1s starts at position 2, length 3
        # Second run of 1s starts at position 6, length 2
        expected = [2, 3, 6, 2]
        assert result == expected
    
    def test_to_rle_all_zeros(self):
        """Test RLE with all zeros (no runs)."""
        bits = np.array([0, 0, 0, 0, 0])
        result = to_rle(bits)
        assert result == []
    
    def test_to_rle_all_ones(self):
        """Test RLE with all ones (single run)."""
        bits = np.array([1, 1, 1, 1, 1])
        result = to_rle(bits)
        assert result == [0, 5]
    
    def test_to_rle_alternating(self):
        """Test RLE with alternating pattern."""
        bits = np.array([1, 0, 1, 0, 1, 0])
        result = to_rle(bits)
        # Three runs of single 1s at positions 0, 2, 4
        expected = [0, 1, 2, 1, 4, 1]
        assert result == expected
    
    def test_to_rle_empty_array(self):
        """Test RLE with empty array."""
        bits = np.array([])
        result = to_rle(bits)
        assert result == []


class TestRefineMasks:
    """Tests for refine_masks function."""
    
    def test_refine_masks_single_mask(self):
        """Test refine_masks with a single mask."""
        # Create a simple mask with one region
        masks = np.zeros((10, 10, 1), dtype=bool)
        masks[2:5, 3:7, 0] = True
        
        rois = np.array([[0, 0, 10, 10]])
        
        refined_masks, refined_rois = refine_masks(masks, rois)
        
        # Check that mask is unchanged
        assert refined_masks.shape == masks.shape
        assert np.array_equal(refined_masks, masks)
        
        # Check that ROI is correctly calculated
        assert refined_rois[0, 0] == 2  # y1
        assert refined_rois[0, 1] == 3  # x1
        assert refined_rois[0, 2] == 4  # y2
        assert refined_rois[0, 3] == 6  # x2
    
    def test_refine_masks_non_overlapping(self):
        """Test refine_masks with non-overlapping masks."""
        masks = np.zeros((10, 10, 2), dtype=bool)
        # First mask in top-left
        masks[1:3, 1:3, 0] = True
        # Second mask in bottom-right
        masks[7:9, 7:9, 1] = True
        
        rois = np.array([[0, 0, 10, 10], [0, 0, 10, 10]])
        
        refined_masks, refined_rois = refine_masks(masks, rois)
        
        # Both masks should remain unchanged (no overlap)
        assert np.sum(refined_masks[:, :, 0]) == 4
        assert np.sum(refined_masks[:, :, 1]) == 4
        
        # Check ROIs are correctly calculated
        assert refined_rois[0, 0] == 1 and refined_rois[0, 2] == 2
        assert refined_rois[1, 0] == 7 and refined_rois[1, 2] == 8
    
    def test_refine_masks_overlapping(self):
        """Test refine_masks with overlapping masks."""
        masks = np.zeros((10, 10, 2), dtype=bool)
        # First mask (smaller area)
        masks[2:5, 2:5, 0] = True
        # Second mask (larger area, overlapping)
        masks[1:7, 1:7, 1] = True
        
        rois = np.array([[0, 0, 10, 10], [0, 0, 10, 10]])
        
        refined_masks, refined_rois = refine_masks(masks, rois)
        
        # The smaller mask should be removed where it overlaps with larger
        # Check that masks don't overlap
        overlap = np.logical_and(refined_masks[:, :, 0], refined_masks[:, :, 1])
        assert not np.any(overlap)
    
    def test_refine_masks_empty_mask(self):
        """Test refine_masks with an empty mask."""
        masks = np.zeros((10, 10, 1), dtype=bool)
        rois = np.array([[0, 0, 10, 10]])
        
        refined_masks, refined_rois = refine_masks(masks, rois)
        
        # Empty mask should remain empty
        assert not np.any(refined_masks)
        # ROI should remain unchanged for empty mask
        assert np.array_equal(refined_rois, rois)
    
    def test_refine_masks_preserves_shape(self):
        """Test that refine_masks preserves mask shape."""
        height, width, num_masks = 20, 20, 3
        masks = np.random.randint(0, 2, (height, width, num_masks), dtype=bool)
        rois = np.zeros((num_masks, 4))
        
        refined_masks, refined_rois = refine_masks(masks, rois)
        
        assert refined_masks.shape == masks.shape
        assert refined_rois.shape == rois.shape
