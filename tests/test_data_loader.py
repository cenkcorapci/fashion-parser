"""Unit tests for data/data_loader.py module."""
import pytest
from unittest.mock import Mock, patch, mock_open, MagicMock
import pandas as pd
import json


class TestDataLoader:
    """Tests for DataLoader class."""
    
    @patch('data.data_loader.pd.read_csv')
    @patch('builtins.open', new_callable=mock_open, read_data='{"info": "test info", "categories": [{"name": "shirt"}, {"name": "pants"}]}')
    def test_data_loader_initialization(self, mock_file, mock_read_csv):
        """Test DataLoader initialization with mocked data."""
        from data.data_loader import DataLoader
        
        # Setup mock CSV data
        mock_df = pd.DataFrame({
            'ImageId': ['img1', 'img1', 'img2'],
            'ClassId': ['1_2', '2_3', '3_4'],
            'EncodedPixels': ['1 2 3 4', '5 6 7 8', '9 10 11 12'],
            'Height': [100, 100, 200],
            'Width': [100, 100, 200]
        })
        mock_read_csv.return_value = mock_df
        
        # Execute
        loader = DataLoader()
        
        # Assert
        assert loader.label_names == ['shirt', 'pants']
        assert mock_file.called
        assert mock_read_csv.called
    
    @patch('data.data_loader.pd.read_csv')
    @patch('builtins.open', new_callable=mock_open, read_data='{"info": "test", "categories": [{"name": "cat1"}]}')
    def test_data_loader_category_id_extraction(self, mock_file, mock_read_csv):
        """Test that CategoryId is correctly extracted from ClassId."""
        from data.data_loader import DataLoader
        
        # Setup
        mock_df = pd.DataFrame({
            'ImageId': ['img1', 'img2'],
            'ClassId': ['10_20', '30_40'],
            'EncodedPixels': ['1 2', '3 4'],
            'Height': [100, 200],
            'Width': [100, 200]
        })
        mock_read_csv.return_value = mock_df
        
        # Execute
        loader = DataLoader()
        
        # Assert - CategoryId should be extracted as first part of ClassId
        assert 'CategoryId' in loader._segment_df.columns
    
    @patch('data.data_loader.pd.read_csv')
    @patch('builtins.open', new_callable=mock_open, read_data='{"info": "test", "categories": [{"name": "cat1"}, {"name": "cat2"}]}')
    def test_data_loader_image_grouping(self, mock_file, mock_read_csv):
        """Test that images are correctly grouped by ImageId."""
        from data.data_loader import DataLoader
        
        # Setup - multiple annotations for same image
        mock_df = pd.DataFrame({
            'ImageId': ['img1', 'img1', 'img2'],
            'ClassId': ['1_2', '2_3', '3_4'],
            'EncodedPixels': ['pixel1', 'pixel2', 'pixel3'],
            'Height': [100, 100, 200],
            'Width': [100, 100, 200]
        })
        mock_read_csv.return_value = mock_df
        
        # Execute
        loader = DataLoader()
        
        # Assert - image_df should group data by ImageId
        assert len(loader.image_df) == 2  # Two unique images
        # First image should have 2 annotations grouped together
        assert isinstance(loader.image_df.iloc[0]['EncodedPixels'], list)
    
    @patch('data.data_loader.pd.read_csv')
    @patch('builtins.open', new_callable=mock_open, read_data='{"info": "test", "categories": []}')
    def test_data_loader_empty_categories(self, mock_file, mock_read_csv):
        """Test DataLoader with empty categories."""
        from data.data_loader import DataLoader
        
        mock_df = pd.DataFrame({
            'ImageId': ['img1'],
            'ClassId': ['1_2'],
            'EncodedPixels': ['1 2'],
            'Height': [100],
            'Width': [100]
        })
        mock_read_csv.return_value = mock_df
        
        # Execute
        loader = DataLoader()
        
        # Assert
        assert loader.label_names == []
    
    @patch('data.data_loader.pd.read_csv')
    @patch('builtins.open', new_callable=mock_open, read_data='{"info": "Fashion 2019", "categories": [{"name": "shirt"}, {"name": "dress"}, {"name": "coat"}]}')
    def test_data_loader_multiple_categories(self, mock_file, mock_read_csv):
        """Test DataLoader with multiple categories."""
        from data.data_loader import DataLoader
        
        mock_df = pd.DataFrame({
            'ImageId': ['img1'],
            'ClassId': ['1_2'],
            'EncodedPixels': ['1 2'],
            'Height': [100],
            'Width': [100]
        })
        mock_read_csv.return_value = mock_df
        
        # Execute
        loader = DataLoader()
        
        # Assert
        assert len(loader.label_names) == 3
        assert 'shirt' in loader.label_names
        assert 'dress' in loader.label_names
        assert 'coat' in loader.label_names
    
    @patch('data.data_loader.pd.read_csv')
    @patch('builtins.open', new_callable=mock_open, read_data='{"info": "test", "categories": [{"name": "cat1"}]}')
    def test_data_loader_size_df(self, mock_file, mock_read_csv):
        """Test that size_df correctly averages Height and Width."""
        from data.data_loader import DataLoader
        
        # Setup - same image with same dimensions
        mock_df = pd.DataFrame({
            'ImageId': ['img1', 'img1', 'img2'],
            'ClassId': ['1_2', '2_3', '3_4'],
            'EncodedPixels': ['p1', 'p2', 'p3'],
            'Height': [100, 100, 200],
            'Width': [150, 150, 250]
        })
        mock_read_csv.return_value = mock_df
        
        # Execute
        loader = DataLoader()
        
        # Assert
        assert 'Height' in loader.size_df.columns
        assert 'Width' in loader.size_df.columns
        assert len(loader.size_df) == 2  # Two unique images
