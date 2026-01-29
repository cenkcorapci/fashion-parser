import numpy as np
import tritonclient.http as httpclient
from PIL import Image
import torch

def infer_fashion(image_path, server_url="localhost:8000"):
    # Preprocess image
    img = Image.open(image_path).convert("RGB")
    img = img.resize((512, 512))
    img_array = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0) # [1, 3, 512, 512]

    # Initialize Triton client
    client = httpclient.InferenceServerClient(url=server_url)

    # Define inputs and outputs
    inputs = [
        httpclient.InferInput("images", img_array.shape, "FP32")
    ]
    inputs[0].set_data_from_numpy(img_array)

    outputs = [
        httpclient.InferRequestedOutput("boxes"),
        httpclient.InferRequestedOutput("labels"),
        httpclient.InferRequestedOutput("scores"),
        httpclient.InferRequestedOutput("masks")
    ]

    # Perform inference
    results = client.infer(model_name="fashion_parser", inputs=inputs, outputs=outputs)

    # Get results
    boxes = results.as_numpy("boxes")
    labels = results.as_numpy("labels")
    scores = results.as_numpy("scores")
    masks = results.as_numpy("masks")

    return {
        "boxes": boxes,
        "labels": labels,
        "scores": scores,
        "masks": masks
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str, help="Path to input image")
    args = parser.parse_args()
    
    try:
        results = infer_fashion(args.image)
        print(f"Detected {len(results['labels'])} objects.")
        print(f"Top class: {results['labels'][0]} with score {results['scores'][0]:.4f}")
    except Exception as e:
        print(f"Inference failed (is Triton running?): {e}")
