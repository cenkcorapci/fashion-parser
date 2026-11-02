"""Export a trained Mask R-CNN to TorchScript for Triton."""
import argparse
import logging
from pathlib import Path

import torch

from fashion_parser.config.settings import IMAGE_SIZE, NUM_CATS
from fashion_parser.models.torch_mrcnn import get_model

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = "model_repository/fashion_parser/1/model.pt"


class TritonWrapper(torch.nn.Module):
    """Adapts the list-based torchvision API to fixed tensor inputs/outputs.

    Only a single image per request is supported (input ``[1, 3, H, W]``).
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        result = self.model([images[0]])[0]
        return result["boxes"], result["labels"], result["scores"], result["masks"]


def export_to_torchscript(weights_path=None, output_path=DEFAULT_OUTPUT, image_size=IMAGE_SIZE):
    model = get_model(NUM_CATS + 1, pretrained=weights_path is None)

    if weights_path:
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        logger.info("Loaded weights from %s", weights_path)
    else:
        logger.warning("No weights given; exporting COCO-initialised heads")

    model.eval()
    dummy_input = torch.randn(1, 3, image_size, image_size)
    traced = torch.jit.trace(TritonWrapper(model), (dummy_input,), strict=False)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    traced.save(output_path)
    logger.info("Model exported to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", help="Path to a .pth state dict")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="TorchScript output path")
    args = parser.parse_args()
    export_to_torchscript(args.weights, args.output)


if __name__ == "__main__":
    main()
