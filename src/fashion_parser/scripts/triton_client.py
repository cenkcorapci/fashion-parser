"""Run inference against a Triton-served fashion parser and optionally visualise it."""
import argparse
import logging

import numpy as np
import tritonclient.http as httpclient
from PIL import Image

from fashion_parser.config.settings import IMAGE_SIZE

logger = logging.getLogger(__name__)

OUTPUT_NAMES = ("boxes", "labels", "scores", "masks")


def preprocess(image_path, image_size=IMAGE_SIZE):
    """Load an image as a float32 ``[1, 3, H, W]`` array in ``[0, 1]``."""
    img = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    array = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return array[None]


def infer_fashion(image_path, server_url="localhost:8000", model_name="fashion_parser"):
    """Return a dict with ``boxes``, ``labels``, ``scores`` and ``masks`` arrays."""
    batch = preprocess(image_path)
    client = httpclient.InferenceServerClient(url=server_url)

    infer_input = httpclient.InferInput("images", batch.shape, "FP32")
    infer_input.set_data_from_numpy(batch)
    outputs = [httpclient.InferRequestedOutput(name) for name in OUTPUT_NAMES]

    results = client.infer(model_name=model_name, inputs=[infer_input], outputs=outputs)
    return {name: results.as_numpy(name) for name in OUTPUT_NAMES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--url", default="localhost:8000", help="Triton HTTP endpoint")
    parser.add_argument("--threshold", type=float, default=0.5, help="Minimum score to report")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        results = infer_fashion(args.image, args.url)
    except Exception as exc:
        raise SystemExit(f"Inference failed (is Triton running?): {exc}")

    keep = results["scores"] >= args.threshold
    logger.info("Detected %d objects (score >= %.2f)", keep.sum(), args.threshold)
    for label, score in zip(results["labels"][keep], results["scores"][keep]):
        logger.info("  class %d  score %.3f", label, score)


if __name__ == "__main__":
    main()
