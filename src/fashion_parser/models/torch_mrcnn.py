"""Mask R-CNN model factory."""
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

MASK_HIDDEN_LAYER = 256


def get_model(num_classes: int, pretrained: bool = True) -> torch.nn.Module:
    """Build a ResNet-50 FPN Mask R-CNN with heads resized for ``num_classes``.

    Args:
        num_classes: Number of classes *including* background.
        pretrained: Start from COCO weights (downloaded on first use).
    """
    model = maskrcnn_resnet50_fpn(weights="DEFAULT" if pretrained else None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, MASK_HIDDEN_LAYER, num_classes
    )
    return model
