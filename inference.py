import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image
import torchvision

def show_prediction(model, img_path, confidence_threshold=0.7, device='cuda'):
    # Load and preprocess image
    img = Image.open(img_path).convert("RGB")
    original_img = np.array(img).copy()

    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((800, 1333)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply transformations
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Run inference
    model.eval()
    with torch.no_grad():
        predictions,losses = model(img_tensor)

    # Move predictions to CPU and convert to numpy
    pred_boxes = predictions[0]['boxes'].cpu().numpy()
    pred_scores = predictions[0]['scores'].cpu().numpy()
    pred_labels = predictions[0]['labels'].cpu().numpy()

    # Filter predictions based on confidence
    mask = pred_scores >= confidence_threshold
    pred_boxes = pred_boxes[mask]
    pred_scores = pred_scores[mask]
    pred_labels = pred_labels[mask]

    # Create figure
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # Display original image
    ax.imshow(original_img)

    # COCO class names (80 classes + background)
    coco_class_names = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    # Plot boxes and labels
    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
        if label <= 80:  # COCO has 80 classes
            # Convert box coordinates to integers
            xmin, ymin, xmax, ymax = box.astype(int)

            # Create rectangle
            rect = plt.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                fill=False, color='red', linewidth=2
            )
            ax.add_patch(rect)

            # Create label text
            text = f"{coco_class_names[label-1]}: {score:.2f}"
            ax.text(
                xmin, ymin - 5, text,
                color='white', fontsize=10,
                bbox=dict(facecolor='red', alpha=0.5, edgecolor='none')
            )

    plt.axis('off')
    plt.show()

# Load trained model (replace with your checkpoint path)
def load_model(checkpoint_path, num_classes=91, device='cuda'):
    model = create_custom_faster_rcnn(
        num_classes=num_classes,
        backbone_arch='vgg16'
    )
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    return model

# Example usage
if __name__ == "__main__":
    # Load your trained model
    model = load_model('/content/checkpoints/frcnn_epoch_2_loss_1.0000.pth')  # Update path

    # Run inference on test image
    show_prediction(model, '/content/train2017/000000000009.jpg', confidence_threshold=0.5)
