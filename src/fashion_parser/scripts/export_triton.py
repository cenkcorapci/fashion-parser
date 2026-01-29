import torch
import os
from fashion_parser.models.torch_mrcnn import get_model
from fashion_parser.config.settings import NUM_CATS

class TritonWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        # images: [B, 3, H, W]
        # torchvision model expects a list of images [3, H, W]
        # images: [B, 3, H, W]. For tracing with B=1, we can just pass [images[0]]
        out = self.model([images[0]])
        # Triton expects fixed number of outputs. 
        # Since we usually serve one image at a time in Triton or batch them,
        # we need to handle the output list.
        # For simplicity, we assume batch size 1 here for the export structure,
        # but Triton can handle dynamic batches if we are careful.
        
        res = out[0]
        # We need to pad or return tensors. 
        # Let's return boxes, labels, scores, masks
        return res["boxes"], res["labels"], res["scores"], res["masks"]

def export_to_torchscript(weights_path=None, output_path="model_repository/fashion_parser/1/model.pt"):
    device = torch.device('cpu')
    num_classes = NUM_CATS + 1
    model = get_model(num_classes)
    
    if weights_path and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded weights from {weights_path}")
    
    model.eval()
    wrapper = TritonWrapper(model)

    # Use tracing for the wrapper since it handles the list conversion
    dummy_input = torch.randn(1, 3, 512, 512)
    traced_model = torch.jit.trace(wrapper, (dummy_input,))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    traced_model.save(output_path)
    print(f"Model successfully exported to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, help="Path to .pth file", default="fashion_mrcnn_pytorch.pth")
    args = parser.parse_args()
    
    export_to_torchscript(args.weights)
