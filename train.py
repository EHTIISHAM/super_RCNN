import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import os
import time
from torchvision.transforms import v2 as transforms
def create_faster_rcnn(num_classes, backbone_name='resnet50', pretrained=True, trainable_layers=3):
    """
    Create Faster R-CNN model with adjustable backbone (ResNet variants with FPN)
    
    Args:
        num_classes (int): Number of output classes (including background)
        backbone_name (str): ResNet backbone name ('resnet18', 'resnet34', 'resnet50', etc.)
        pretrained (bool): Use pretrained weights for backbone
        trainable_layers (int): Number of backbone layers to train (0-5)
        
    Returns:
        FasterRCNN model
    """
    # Create backbone with FPN
    backbone = resnet_fpn_backbone(
        backbone_name=backbone_name,
        pretrained=pretrained,
        trainable_layers=trainable_layers
    )
    
    # Create Faster R-CNN model
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        min_size=800,       # Minimum input size
        max_size=1333       # Maximum input size
    )
    
    return model

def create_custom_backbone_model(num_classes, backbone_arch='vgg16', pretrained=True):
    """
    Create Faster R-CNN with custom backbone (supports VGG, DenseNet, EfficientNet, MobileNet)
    
    Args:
        num_classes (int): Number of output classes
        backbone_arch (str): Backbone architecture name
        pretrained (bool): Use pretrained weights
        
    Returns:
        FasterRCNN model
    """
    backbone = None
    anchor_sizes = ((32, 64, 128, 256, 512),)  # Default anchor sizes
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

    # Create custom backbone
    if backbone_arch == 'vgg16':
        backbone = torchvision.models.vgg16(pretrained=pretrained).features
        backbone.out_channels = 512
        
    elif backbone_arch == 'densenet121':
        backbone = torchvision.models.densenet121(pretrained=pretrained).features
        backbone.out_channels = 1024
        anchor_sizes = ((64, 128, 256, 512, 1024),)  # Adjusted for DenseNet's deeper features
        
    elif backbone_arch.startswith('efficientnet'):
        if backbone_arch == 'efficientnet_b0':
            backbone = torchvision.models.efficientnet_b0(pretrained=pretrained).features
            backbone.out_channels = 1280
        elif backbone_arch == 'efficientnet_b4':
            backbone = torchvision.models.efficientnet_b4(pretrained=pretrained).features
            backbone.out_channels = 1792
        anchor_sizes = ((16, 32, 64, 128, 256),)  # Smaller anchors for EfficientNet's high-resolution features
        
    elif backbone_arch.startswith('mobilenet'):
        if 'v3_large' in backbone_arch:
            backbone = torchvision.models.mobilenet_v3_large(pretrained=pretrained).features
            backbone.out_channels = 960
        elif 'v3_small' in backbone_arch:
            backbone = torchvision.models.mobilenet_v3_small(pretrained=pretrained).features
            backbone.out_channels = 576
        anchor_sizes = ((16, 32, 64, 128, 256),)  # Smaller anchors for mobile-oriented networks
        
    else:
        raise ValueError(f"Unsupported backbone: {backbone_arch}")

    # Define anchor generator with updated sizes
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

    # Define ROI pooler (adjust based on backbone)
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],  # Use only the final feature map
        output_size=7,
        sampling_ratio=2
    )

    # Create model
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=600,  # Can adjust based on backbone
        max_size=1000   # Can adjust based on backbone
    )
    
    return model


# Dataset setup
def get_transform(train):
    transform_list = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.RandomHorizontalFlip(0.5 if train else 0),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Resize((800, 1333), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    return transform_list

class CocoDataset(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms=None):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
            areas.append(ann["area"])
            iscrowd.append(ann.get("iscrowd", 0))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id]),
            "area": areas,
            "iscrowd": iscrowd
        }

        if self._transforms:
            try:
                img, target = self._transforms(img, target)
                # Post-transform validation
                if len(target["boxes"]) == 0:
                    return None
                # Ensure box validity after transforms
                if (target["boxes"][:, 2:] <= target["boxes"][:, :2]).any():
                    return None
            except Exception as e:
                print(f"Transform error: {e}")
                return None

        return img, target

def collate_fn(batch):
    # Filter out None entries (empty images)
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None, None  # Handle in training loop
    return tuple(zip(*batch))

# Training function
def train_model(
    model: FasterRCNN,
    train_dataset,
    val_dataset,
    num_epochs=10,
    batch_size=4,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005,
    checkpoint_dir='checkpoints',
    device='cuda'
):
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )

    # Optimizer setup
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )

    # Move model to device
    model = model.to(device)
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()
        
        # Initialize metrics
        epoch_loss = 0.0
        last_loss = 0.0
        start_time = time.time()
        
        # Training phase
        with tqdm(train_loader, unit="batch") as tepoch:
            for batch_idx, batch in enumerate(tepoch):
                images, targets = batch

                if batch is None or (batch[0] is None and batch[1] is None):
                    continue
                if len(images) == 0 or any(len(t["boxes"]) == 0 for t in targets):
                    continue
                # Move data to device
                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # Forward pass
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                # Backward pass
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                # Update metrics
                epoch_loss += losses.item()
                last_loss = losses.item()
                
                # Update progress bar
                tepoch.set_postfix({
                    'loss': f"{last_loss:.4f}",
                    'avg_loss': f"{epoch_loss/(batch_idx+1):.4f}"
                })

        # Calculate epoch metrics
        epoch_loss /= len(train_loader)
        epoch_time = time.time() - start_time
        
        print(f"Train Loss: {epoch_loss:.4f} | Time: {epoch_time:.2f}s")

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad(), tqdm(val_loader, unit="batch") as vepoch:
            for images, targets in vepoch:
                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # Forward pass
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()

                vepoch.set_postfix({
                    'val_loss': f"{losses.item():.4f}"
                })

        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"frcnn_epoch_{epoch+1}_loss_{val_loss:.4f}.pth"
        )
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': epoch_loss,
            'val_loss': val_loss,
        }, checkpoint_path)
        
    return model

# Example usage
if __name__ == "__main__":
    # Crea
    # Dataset paths
    train_root = "train2017"
    train_ann = "annotations/instances_train2017.json"
    val_root = "val2017"
    val_ann = "annotations/instances_val2017.json"

    # Create datasets
    train_ds = CocoDataset(
        img_folder=train_root,
        ann_file=train_ann,
        transforms=get_transform(train=True)
    )
    val_ds = CocoDataset(
        img_folder=val_root,
        ann_file=val_ann,
        transforms=get_transform(train=False)
    )
    # Create VGG16 based model
    model = create_custom_backbone_model(
        num_classes=91,
        backbone_arch='mobilenet_v3_large'
    )
    
    # Train the model
    trained_model = train_model(
        model,
        train_ds,
        val_ds,
        num_epochs=1,
        batch_size=4,
        lr=0.005,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
