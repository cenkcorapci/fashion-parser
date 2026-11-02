"""Train Mask R-CNN on the iMaterialist FGVC6 dataset."""
import argparse
import logging

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from fashion_parser.config.settings import NUM_CATS, RANDOM_STATE
from fashion_parser.data.loader import DataLoader as FashionDataLoader
from fashion_parser.data.torch_dataset import TorchFashionDataset
from fashion_parser.models.torch_mrcnn import get_model

logger = logging.getLogger(__name__)


def collate_fn(batch):
    return tuple(zip(*batch))


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0.0
    pbar = tqdm(data_loader, desc=f"Epoch {epoch}")
    for images, targets in pbar:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss = sum(model(images, targets).values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
    return total_loss / len(data_loader)


@torch.no_grad()
def evaluate_loss(model, data_loader, device):
    """Average loss on a validation set (model stays in train mode to emit losses)."""
    model.train()
    total_loss = 0.0
    for images, targets in data_loader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        total_loss += sum(model(images, targets).values()).item()
    return total_loss / max(len(data_loader), 1)


def train_model(epochs=10, val_split=0.1, batch_size=2, lr=0.005, output="fashion_mrcnn_pytorch.pth"):
    device = get_device()
    logger.info("Using device: %s", device)

    loader = FashionDataLoader()
    train_df, val_df = train_test_split(loader.image_df, random_state=RANDOM_STATE, test_size=val_split)

    train_loader = DataLoader(
        TorchFashionDataset(train_df, loader.label_names),
        batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        TorchFashionDataset(val_df, loader.label_names),
        batch_size=1, shuffle=False, collate_fn=collate_fn,
    )

    model = get_model(NUM_CATS + 1).to(device)  # +1 for background
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        val_loss = evaluate_loss(model, val_loader, device)
        logger.info("Epoch %d: train loss %.4f, val loss %.4f", epoch, train_loss, val_loss)
        torch.save(model.state_dict(), output)

    logger.info("Weights saved to %s", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--output", default="fashion_mrcnn_pytorch.pth")
    args = parser.parse_args()
    train_model(args.epochs, args.val_split, args.batch_size, args.lr, args.output)


if __name__ == "__main__":
    main()
