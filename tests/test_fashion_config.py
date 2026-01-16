"""Unit tests for commons/fashion_config.py module."""
import pytest
from unittest.mock import patch

from commons.fashion_config import FashionConfig, InferenceConfig


class TestFashionConfig:
    """Tests for FashionConfig class."""
    
    def test_fashion_config_initialization(self):
        """Test FashionConfig initialization with default values."""
        config = FashionConfig()
        
        assert config.NAME == "fashion_resnet_101"
        assert config.GPU_COUNT == 1
        assert config.IMAGES_PER_GPU == 2
        assert config.BACKBONE == 'resnet101'
    
    def test_fashion_config_num_classes(self):
        """Test that NUM_CLASSES includes background class."""
        config = FashionConfig()
        
        # NUM_CLASSES should be NUM_CATS + 1 (for background)
        from commons.config import NUM_CATS
        assert config.NUM_CLASSES == NUM_CATS + 1
    
    def test_fashion_config_image_settings(self):
        """Test image dimension settings."""
        config = FashionConfig()
        
        from commons.config import IMAGE_SIZE
        assert config.IMAGE_MIN_DIM == IMAGE_SIZE
        assert config.IMAGE_MAX_DIM == IMAGE_SIZE
        assert config.IMAGE_RESIZE_MODE == 'none'
    
    def test_fashion_config_steps_per_epoch(self):
        """Test steps per epoch configuration."""
        config = FashionConfig()
        
        # Should be set to a large number to use all data
        assert config.STEPS_PER_EPOCH == 10000000
        assert config.VALIDATION_STEPS == 100000
    
    def test_fashion_config_rpn_anchor_scales(self):
        """Test RPN anchor scales configuration."""
        config = FashionConfig()
        
        expected_scales = (16, 32, 64, 128, 256)
        assert config.RPN_ANCHOR_SCALES == expected_scales
    
    def test_fashion_config_backbone(self):
        """Test backbone configuration."""
        config = FashionConfig()
        
        assert config.BACKBONE == 'resnet101'


class TestInferenceConfig:
    """Tests for InferenceConfig class."""
    
    def test_inference_config_initialization(self):
        """Test InferenceConfig initialization."""
        config = InferenceConfig()
        
        # Should inherit from FashionConfig
        assert config.NAME == "fashion_resnet_101"
        assert config.BACKBONE == 'resnet101'
    
    def test_inference_config_gpu_settings(self):
        """Test that inference config has correct GPU settings."""
        config = InferenceConfig()
        
        assert config.GPU_COUNT == 1
        assert config.IMAGES_PER_GPU == 1
    
    def test_inference_config_inherits_from_fashion_config(self):
        """Test that InferenceConfig properly inherits from FashionConfig."""
        config = InferenceConfig()
        
        # Check inherited properties
        from commons.config import IMAGE_SIZE, NUM_CATS
        assert config.IMAGE_MIN_DIM == IMAGE_SIZE
        assert config.IMAGE_MAX_DIM == IMAGE_SIZE
        assert config.NUM_CLASSES == NUM_CATS + 1
    
    def test_inference_config_overrides_images_per_gpu(self):
        """Test that InferenceConfig overrides IMAGES_PER_GPU correctly."""
        fashion_config = FashionConfig()
        inference_config = InferenceConfig()
        
        # FashionConfig has IMAGES_PER_GPU = 2
        assert fashion_config.IMAGES_PER_GPU == 2
        # InferenceConfig should override to 1
        assert inference_config.IMAGES_PER_GPU == 1
    
    def test_inference_and_fashion_config_same_name(self):
        """Test that both configs have the same name."""
        fashion_config = FashionConfig()
        inference_config = InferenceConfig()
        
        assert fashion_config.NAME == inference_config.NAME


class TestConfigConstants:
    """Tests for configuration constants used by config classes."""
    
    def test_image_size_is_positive(self):
        """Test that IMAGE_SIZE is a positive integer."""
        from commons.config import IMAGE_SIZE
        
        assert isinstance(IMAGE_SIZE, int)
        assert IMAGE_SIZE > 0
    
    def test_num_cats_is_positive(self):
        """Test that NUM_CATS is a positive integer."""
        from commons.config import NUM_CATS
        
        assert isinstance(NUM_CATS, int)
        assert NUM_CATS > 0
    
    def test_config_constants_relationship(self):
        """Test relationship between config constants."""
        config = FashionConfig()
        from commons.config import NUM_CATS
        
        # NUM_CLASSES should always be NUM_CATS + 1
        assert config.NUM_CLASSES == NUM_CATS + 1
