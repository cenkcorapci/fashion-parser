import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from fashion_parser.data.loader import DataLoader as FashionDataLoader
from fashion_parser.data.torch_dataset import TorchFashionDataset
from fashion_parser.models.torch_mrcnn import get_model
from fashion_parser.config.settings import RANDOM_STATE, NUM_CATS
from tqdm import tqdm

def collate_fn(batch):
    return tuple(zip(*batch))

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0
    
    pbar = tqdm(data_loader, desc=f'Epoch {epoch}')
    for images, targets in pbar:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        pbar.set_postfix(loss=losses.item())
        
    return total_loss / len(data_loader)

def train_model(epochs=10, val_split=0.1):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load data
    loader = FashionDataLoader()
    df = loader.image_df
    
    train_df, val_df = train_test_split(df, random_state=RANDOM_STATE, test_size=val_split)
    
    train_dataset = TorchFashionDataset(train_df, loader.label_names)
    val_dataset = TorchFashionDataset(val_df, loader.label_names)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Initialize model
    num_classes = NUM_CATS + 1 # background + categories
    model = get_model(num_classes)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    for epoch in range(epochs):
        avg_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
        
    # Save model
    torch.save(model.state_dict(), "fashion_mrcnn_pytorch.pth")

if __name__ == "__main__":
    train_model(epochs=2)
