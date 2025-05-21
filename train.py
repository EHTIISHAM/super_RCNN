import torch
import torch.nn as nn
import torchvision
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork
from torchvision.ops import boxes as box_ops
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import os
import time
from torchvision.transforms import v2 as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class CustomFPN(nn.Module):
    """Feature Pyramid Network for any backbone"""
    def __init__(self, backbone, in_channels_list, out_channels=256):
        super().__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()

        # Add 1x1 conv for channel reduction
        for in_channels in in_channels_list:
            self.inner_blocks.append(nn.Conv2d(in_channels, out_channels, 1))
            self.layer_blocks.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))

    def forward(self, x):
        # x = OrderedDict of feature maps from backbone
        last_inner = self.inner_blocks[-1](list(x.values())[-1])
        results = [self.layer_blocks[-1](last_inner)]

        for idx in range(len(x)-2, -1, -1):
            inner_lateral = self.inner_blocks[idx](list(x.values())[idx])
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = nn.functional.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.layer_blocks[idx](last_inner))

        return {str(i): v for i, v in enumerate(results)}

class CustomBackbone(nn.Module):
    """Wrapper for any backbone with FPN and 1x1 conv"""
    def __init__(self, backbone, return_layers, in_channels_list, fpn_out_channels=256):
        super().__init__()
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = CustomFPN(backbone, in_channels_list, fpn_out_channels)
        self.out_channels = fpn_out_channels

    def forward(self, x):
        x = self.body(x)
        return self.fpn(x)

class FeatureExtractingRoIHeads(RoIHeads):
    def forward(self, features, proposals, image_shapes, targets=None):
        """
        Modified forward pass to include features in outputs
        """
        # Original processing
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)

        # Store features before final prediction
        features_before_predictor = box_features.clone()

        # Get predictions
        class_logits, box_regression = self.box_predictor(box_features)

        # Post-process detections
        result, losses = self.postprocess_detections(
            class_logits, box_regression, proposals, image_shapes
        )

        # Add features to results
        if not self.training:
            for res in result:
                res['features'] = features_before_predictor
        return result, losses

class CustomFasterRCNN(nn.Module):
    def __init__(self, backbone, num_classes,backbone_arch, min_size=800, max_size=1333):
        super().__init__()
        self.backbone = backbone
        self.transform = GeneralizedRCNNTransform(min_size, max_size,
                                                 [0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])

        # RPN Configuration
        if backbone_arch.startswith('resnet'):
          anchor_sizes = ((32,), (64,), (128,), (256,))
          aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        elif backbone_arch == 'vgg16':
          anchor_sizes = ((32,), (64,), (128,))
          aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        else:
          #default
          anchor_sizes = ((32,), (64,), (128,), (256,))
          aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        rpn_head = RPNHead(backbone.out_channels, len(aspect_ratios[0]))

        self.rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            fg_iou_thresh=0.7,
            bg_iou_thresh=0.3,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n={'training': 2000, 'testing': 1000},
            post_nms_top_n={'training': 2000, 'testing': 1000},
            nms_thresh=0.7
        )

        # ROI Heads Configuration
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2
        )

        resolution = box_roi_pool.output_size[0]
        representation_size = 1024

        box_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone.out_channels * resolution ** 2, representation_size),
            nn.ReLU(),
            nn.Linear(representation_size, representation_size))

        box_predictor = FastRCNNPredictor(representation_size, num_classes)

        self.roi_heads = RoIHeads(
            # Box parameters
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=512,
            positive_fraction=0.25,
            bbox_reg_weights=None,
            score_thresh=0.05,
            nms_thresh=0.5,
            detections_per_img=100
        )

    def forward(self, images, targets=None):
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)

        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals,
                                                   images.image_sizes, targets)

        losses = {}
        losses.update(proposal_losses)
        losses.update(detector_losses)
        if not self.training:
            detections = self.transform.postprocess(
                detections, images.image_sizes, original_image_sizes
            )

        return detections,losses


def create_custom_faster_rcnn(num_classes, backbone_arch='resnet50', pretrained=True):
    # Create base backbone
    if backbone_arch.startswith('resnet'):
        backbone = torchvision.models.__dict__[backbone_arch](pretrained=pretrained)
        return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
        in_channels_list = [256, 512, 1024, 2048][:4]
    elif backbone_arch == 'vgg16':
        backbone = torchvision.models.vgg16(pretrained=pretrained).features
        return_layers = {'16': '0', '23': '1', '30': '2'}
        in_channels_list = [256, 512, 512]
    else:  # Add other architectures similarly
        raise ValueError(f"Unsupported backbone: {backbone_arch}")

    # Wrap with FPN and 1x1 conv
    backbone = CustomBackbone(backbone, return_layers, in_channels_list)

    return CustomFasterRCNN(backbone, num_classes,backbone_arch)


# Dataset setup
def get_transform(train):
    transform_list = transforms.Compose([
        transforms.ToImage(),  # Converts to tensor and handles PIL/Numpy inputs
        transforms.ToDtype(torch.float32, scale=True),

        # Convert RGB to BGR by reversing channels
        transforms.Lambda(lambda x: x[[2, 1, 0],]),  # Channel order: BGR

        # Continue with standard augmentations
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),

        # Adjust normalization for BGR format (original ImageNet RGB mean/std reversed)
        transforms.Normalize(
            mean=[0.406, 0.456, 0.485],  # BGR mean (original RGB [0.485, 0.456, 0.406])
            std=[0.225, 0.224, 0.229]    # BGR std (original RGB [0.229, 0.224, 0.225])
        )
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

def show_prediction(model, dataset, device, num_images=2):
    model.eval()
    fig, axs = plt.subplots(1, num_images, figsize=(16, 8))
    if num_images == 1:
        axs = [axs]

    for i in range(num_images):
        image, _ = dataset[i]
        img_tensor = image.to(device).unsqueeze(0)

        with torch.no_grad():
            preds = model(img_tensor)[0]

        img_np = image.permute(1, 2, 0).cpu().numpy()
        axs[i].imshow(img_np)
        axs[i].axis("off")

        for box in preds["boxes"].cpu():
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            axs[i].add_patch(rect)

    plt.tight_layout()
    plt.show()

# Training function
def train_model(
    model, train_dataset, val_dataset,
    num_epochs=10, batch_size=4, lr=0.005,
    momentum=0.9, weight_decay=0.0005,
    checkpoint_dir='checkpoints', device='cuda'
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

    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    model.to(device)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, (images, targets) in enumerate(pbar, start=1):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            detections,loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss
            avg_loss = epoch_loss / batch_idx
            pbar.set_postfix({'batch_loss': f"{batch_loss:.4f}", 'avg_loss': f"{avg_loss:.4f}"})

        avg_loss = epoch_loss / len(train_loader)
        print(f"Train Loss: {avg_loss:.4f} | Time: {time.time() - start_time:.2f}s")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validation"):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
                model.train() # Force the model to return loss during validation as in eval model just ignores it
                detection,loss_dict = model(images, targets)
                model.eval() # return to eval state
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # Checkpoint
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_loss,
            'val_loss': avg_val_loss
        }, os.path.join(checkpoint_dir, f"frcnn_epoch_{epoch+1}.pth"))
        # Show predictions
        if (epoch + 1) % show_every_n_epochs == 0:
            print(f"Showing predictions at epoch {epoch+1}")
            show_prediction(model, val_dataset, device)
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
    model = create_custom_faster_rcnn(
        num_classes=91,
        backbone_arch='vgg16'
    )

    # Train the model
    trained_model = train_model(
        model,
        train_ds,
        val_ds,
        num_epochs=1,
        batch_size=6,
        lr=0.0001,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
