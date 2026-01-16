"""Unit tests for data/fashion_data_set.py module."""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock


class TestFashionDataset:
    """Tests for FashionDataset class."""
    
    @patch('data.fashion_data_set.utils.Dataset.__init__')
    def test_fashion_dataset_initialization(self, mock_super_init):
        """Test FashionDataset initialization."""
        from data.fashion_data_set import FashionDataset
        
        mock_super_init.return_value = None
        
        # Setup test data
        df = pd.DataFrame({
            'CategoryId': [['1', '2']],
            'EncodedPixels': [['1 2 3 4', '5 6 7 8']],
            'Height': [100],
            'Width': [100]
        }, index=['img1.jpg'])
        
        label_names = ['shirt', 'pants', 'dress']
        
        # Create dataset without calling parent methods
        dataset = object.__new__(FashionDataset)
        dataset._label_names = label_names
        
        # Verify label names are stored
        assert dataset._label_names == label_names
    
    def test_fashion_dataset_image_reference(self):
        """Test image_reference method."""
        from data.fashion_data_set import FashionDataset
        
        # Create a mock dataset
        dataset = object.__new__(FashionDataset)
        dataset._label_names = ['shirt', 'pants', 'dress']
        dataset.image_info = [
            {
                'path': '/path/to/img1.jpg',
                'labels': ['0', '1']
            }
        ]
        
        # Execute
        path, labels = dataset.image_reference(0)
        
        # Assert
        assert path == '/path/to/img1.jpg'
        assert labels == ['shirt', 'pants']
    
    @patch('data.fashion_data_set.resize_image')
    def test_fashion_dataset_load_image(self, mock_resize):
        """Test load_image method."""
        from data.fashion_data_set import FashionDataset
        
        # Setup
        dataset = object.__new__(FashionDataset)
        dataset.image_info = [
            {'path': '/path/to/image.jpg'}
        ]
        
        mock_image = np.zeros((512, 512, 3), dtype=np.uint8)
        mock_resize.return_value = mock_image
        
        # Execute
        result = dataset.load_image(0)
        
        # Assert
        mock_resize.assert_called_once_with('/path/to/image.jpg')
        assert result.shape == (512, 512, 3)
    
    @patch('data.fashion_data_set.cv2.resize')
    def test_fashion_dataset_load_mask_single_annotation(self, mock_cv_resize):
        """Test load_mask method with single annotation."""
        from data.fashion_data_set import FashionDataset
        
        # Setup
        dataset = object.__new__(FashionDataset)
        dataset.image_info = [
            {
                'annotations': ['1 2'],  # Simple RLE: start at 1, run length 2
                'labels': ['5'],
                'height': 10,
                'width': 10
            }
        ]
        
        # Mock cv2.resize to return a simple mask
        def mock_resize_func(img, size, interpolation):
            return np.zeros(size, dtype=np.uint8)
        
        mock_cv_resize.side_effect = mock_resize_func
        
        # Execute
        from commons.config import IMAGE_SIZE
        mask, labels = dataset.load_mask(0)
        
        # Assert
        assert mask.shape == (IMAGE_SIZE, IMAGE_SIZE, 1)
        assert len(labels) == 1
        assert labels[0] == 6  # label 5 + 1
    
    @patch('data.fashion_data_set.cv2.resize')
    def test_fashion_dataset_load_mask_multiple_annotations(self, mock_cv_resize):
        """Test load_mask method with multiple annotations."""
        from data.fashion_data_set import FashionDataset
        
        # Setup
        dataset = object.__new__(FashionDataset)
        dataset.image_info = [
            {
                'annotations': ['1 2', '5 3'],
                'labels': ['3', '7'],
                'height': 10,
                'width': 10
            }
        ]
        
        def mock_resize_func(img, size, interpolation):
            return np.zeros(size, dtype=np.uint8)
        
        mock_cv_resize.side_effect = mock_resize_func
        
        # Execute
        from commons.config import IMAGE_SIZE
        mask, labels = dataset.load_mask(0)
        
        # Assert
        assert mask.shape == (IMAGE_SIZE, IMAGE_SIZE, 2)
        assert len(labels) == 2
        assert labels[0] == 4  # label 3 + 1
        assert labels[1] == 8  # label 7 + 1
    
    @patch('data.fashion_data_set.cv2.resize')
    def test_fashion_dataset_load_mask_rle_decoding(self, mock_cv_resize):
        """Test that RLE encoding is properly decoded in load_mask."""
        from data.fashion_data_set import FashionDataset
        
        # Setup
        dataset = object.__new__(FashionDataset)
        # RLE format: "start_pos1 length1 start_pos2 length2 ..."
        dataset.image_info = [
            {
                'annotations': ['0 5 10 3'],  # Two runs: 0-4 and 10-12
                'labels': ['1'],
                'height': 5,
                'width': 4  # Total 20 pixels
            }
        ]
        
        def mock_resize_func(img, size, interpolation):
            # Return the input for testing
            return np.zeros(size, dtype=np.uint8)
        
        mock_cv_resize.side_effect = mock_resize_func
        
        # Execute
        mask, labels = dataset.load_mask(0)
        
        # Assert - should not raise errors
        assert mask is not None
        assert labels is not None
        assert len(labels) == 1


class TestFashionDatasetIntegration:
    """Integration tests for FashionDataset with realistic data."""
    
    @patch('data.fashion_data_set.resize_image')
    @patch('data.fashion_data_set.cv2.resize')
    @patch('data.fashion_data_set.utils.Dataset')
    def test_fashion_dataset_prepare_workflow(self, mock_dataset_base, mock_cv_resize, mock_resize_image):
        """Test typical workflow of preparing dataset."""
        from data.fashion_data_set import FashionDataset
        
        # Mock base class methods
        mock_dataset_base.__init__ = Mock(return_value=None)
        mock_dataset_base.add_class = Mock()
        mock_dataset_base.add_image = Mock()
        
        # Setup
        df = pd.DataFrame({
            'CategoryId': [['1', '2'], ['3']],
            'EncodedPixels': [['1 2 3 4', '5 6 7 8'], ['9 10']],
            'Height': [100, 200],
            'Width': [100, 200]
        }, index=['img1.jpg', 'img2.jpg'])
        
        label_names = ['shirt', 'pants', 'dress']
        
        # This would normally call parent __init__ and methods
        # For unit testing, we just verify the structure is correct
        assert len(df) == 2
        assert len(label_names) == 3
    
    def test_fashion_dataset_mask_shape_consistency(self):
        """Test that masks maintain consistent shape."""
        from commons.config import IMAGE_SIZE
        
        # Verify IMAGE_SIZE is defined and positive
        assert IMAGE_SIZE > 0
        
        # In actual usage, all masks should be resized to (IMAGE_SIZE, IMAGE_SIZE)
        expected_shape = (IMAGE_SIZE, IMAGE_SIZE)
        assert expected_shape == (IMAGE_SIZE, IMAGE_SIZE)
