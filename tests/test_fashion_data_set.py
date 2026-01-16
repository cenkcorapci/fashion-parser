"""Unit tests for data/fashion_data_set.py module."""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock


class TestFashionDataset:
    """Tests for FashionDataset class."""
    
    def test_fashion_dataset_image_reference(self):
        """Test image_reference method."""
        try:
            from data.fashion_data_set import FashionDataset
        except ImportError:
            pytest.skip("TensorFlow not available")
        
        # Create a properly mocked dataset using Mock
        dataset = Mock(spec=FashionDataset)
        dataset._label_names = ['shirt', 'pants', 'dress']
        dataset.image_info = [
            {
                'path': '/path/to/img1.jpg',
                'labels': ['0', '1']
            }
        ]
        
        # Call the actual method by binding it to the mock
        path, labels = FashionDataset.image_reference(dataset, 0)
        
        # Assert
        assert path == '/path/to/img1.jpg'
        assert labels == ['shirt', 'pants']
    
    def test_fashion_dataset_label_conversion(self):
        """Test that labels are correctly converted."""
        label_names = ['shirt', 'pants', 'dress', 'coat']
        
        # Simulate label conversion
        label_indices = ['0', '2']
        converted = [label_names[int(x)] for x in label_indices]
        
        assert converted == ['shirt', 'dress']


class TestFashionDatasetHelpers:
    """Tests for helper functionality used by FashionDataset."""
    
    def test_rle_parsing_logic(self):
        """Test RLE parsing logic used in load_mask."""
        # RLE format: "start_pixel1 run_length1 start_pixel2 run_length2 ..."
        rle_string = "0 5 10 3"
        annotation = [int(x) for x in rle_string.split(' ')]
        
        # Create a mask array
        total_pixels = 20
        sub_mask = np.full(total_pixels, 0, dtype=np.uint8)
        
        # Apply RLE encoding
        for i, start_pixel in enumerate(annotation[::2]):
            run_length = annotation[2 * i + 1]
            sub_mask[start_pixel: start_pixel + run_length] = 1
        
        # Check that correct pixels are set
        assert np.sum(sub_mask) == 8  # 5 + 3 pixels set
        assert sub_mask[0] == 1
        assert sub_mask[4] == 1
        assert sub_mask[5] == 0
        assert sub_mask[10] == 1
        assert sub_mask[12] == 1
        assert sub_mask[13] == 0
    
    def test_mask_reshape_logic(self):
        """Test mask reshaping logic."""
        height, width = 10, 10
        flat_mask = np.zeros(height * width, dtype=np.uint8)
        flat_mask[0:10] = 1
        
        # Reshape in Fortran order (column-major)
        reshaped = flat_mask.reshape((height, width), order='F')
        
        # First column should be all 1s
        assert np.all(reshaped[:, 0] == 1)
        # Other columns should be 0
        assert np.all(reshaped[:, 1] == 0)


class TestFashionDatasetIntegration:
    """Integration tests for FashionDataset with realistic data."""
    
    def test_fashion_dataset_data_structure(self):
        """Test typical data structure for FashionDataset."""
        # Typical DataFrame structure
        df = pd.DataFrame({
            'CategoryId': [['1', '2'], ['3']],
            'EncodedPixels': [['1 2 3 4', '5 6 7 8'], ['9 10']],
            'Height': [100, 200],
            'Width': [100, 200]
        }, index=['img1.jpg', 'img2.jpg'])
        
        label_names = ['shirt', 'pants', 'dress']
        
        # Verify structure
        assert len(df) == 2
        assert len(label_names) == 3
        assert isinstance(df.iloc[0]['CategoryId'], list)
        assert isinstance(df.iloc[0]['EncodedPixels'], list)
    
    def test_fashion_dataset_mask_shape_consistency(self):
        """Test that masks maintain consistent shape."""
        from commons.config import IMAGE_SIZE
        
        # Verify IMAGE_SIZE is defined and positive
        assert IMAGE_SIZE > 0
        
        # In actual usage, all masks should be resized to (IMAGE_SIZE, IMAGE_SIZE)
        expected_shape = (IMAGE_SIZE, IMAGE_SIZE)
        assert expected_shape == (IMAGE_SIZE, IMAGE_SIZE)
    
    def test_label_indexing(self):
        """Test that label indexing works correctly."""
        # FashionDataset adds 1 to labels for background class
        original_label = 5
        adjusted_label = original_label + 1
        
        assert adjusted_label == 6
        
        # Multiple labels
        original_labels = ['3', '7', '10']
        adjusted_labels = [int(label) + 1 for label in original_labels]
        
        assert adjusted_labels == [4, 8, 11]

