"""
Inference script for trained SSD300 model on animal detection.
Loads trained weights and performs object detection on images.
Outputs results in clean dictionary/JSON format for downstream processing.
Optimized for COCO dataset format with automatic class name mapping.
"""

import os
from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional
import json

import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image

from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDClassificationHead
import torchvision.transforms as transforms


# ============================================================================
# ======================== CONFIGURATION ====================================
# ============================================================================

NUM_CLASSES = 8  # 6 animal classes + 1 background + 1 null
CONFIDENCE_THRESHOLD = 0.9  # Only keep detections with score >= threshold

MODEL_PATH = './checkpoints/best_ssd.pth'
COCO_ANN_FILE = './animal_dataset/test/_annotations.coco.json'  # To load class names

# **CRITICAL:** Explicitly use CUDA if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# ======================== UTILITY FUNCTIONS ================================
# ============================================================================


def load_class_names_from_coco(ann_file: str) -> Dict[int, str]:
    """
    Load class index -> class name mapping from COCO JSON file.
    
    Args:
        ann_file: Path to COCO JSON annotation file
    
    Returns:
        Dictionary: {1: 'cat', 2: 'dog', ...} (1-indexed; 0 is background)
    """
    if not os.path.exists(ann_file):
        print(f"Warning: COCO annotation file not found: {ann_file}")
        print("Using default class names")
        return {i: f'class_{i}' for i in range(1, NUM_CLASSES)}
    
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    # Build 1-indexed mapping (0 is reserved for background)
    idx_to_class = {}
    for idx, cat in enumerate(sorted(coco_data.get('categories', []), key=lambda x: x['id']), start=1):
        idx_to_class[idx] = cat['name']
    
    return idx_to_class


def _get_ssd_head_specs(model: nn.Module) -> tuple[list[int], list[int]]:
    """
    Extract the channel layout required to rebuild the SSD classification head.

    torchvision's SSD implementation exposes the per-feature-map conv layers on
    ``classification_head.module_list`` and the anchor counts on the model's
    ``anchor_generator``. This is much more stable than relying on internal
    attributes that vary across torchvision versions.
    """
    cls_head = model.head.classification_head

    if hasattr(cls_head, "module_list"):
        in_channels = [layer.in_channels for layer in cls_head.module_list]
    elif hasattr(cls_head, "extra_layers"):
        in_channels = [layer[0].in_channels for layer in cls_head.extra_layers]
    else:
        raise AttributeError(
            "Unable to infer SSD classification head layout from the current torchvision version."
        )

    if hasattr(model.anchor_generator, "num_anchors_per_location"):
        num_anchors = list(model.anchor_generator.num_anchors_per_location())
    else:
        num_anchors = [layer.out_channels // cls_head.num_columns for layer in cls_head.module_list]

    return in_channels, num_anchors


def modify_ssd_head(model: nn.Module, num_classes: int) -> nn.Module:
    """
    Modify the classification head of SSD300 to match the number of classes.
    (Same as in train.py)
    """
    in_channels, num_anchors = _get_ssd_head_specs(model)
    
    new_classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
    )
    
    model.head.classification_head = new_classification_head
    return model


def load_model(model_path: str, num_classes: Optional[int] = None, device='cuda') -> nn.Module:
    """
    Load trained SSD300 model.
    
    Args:
        model_path: Path to saved .pth file
        num_classes: Number of classes (must match training config)
        device: 'cuda' or 'cpu'
    
    Returns:
        Loaded model in eval mode
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load pre-trained SSD300
    model = ssd300_vgg16(weights='DEFAULT')
    
    # Modify head to match training config
    if num_classes is None:
        # Infer the number of classes from the annotation file used for training.
        num_classes = len(load_class_names_from_coco(COCO_ANN_FILE)) + 1

    model = modify_ssd_head(model, num_classes=num_classes)
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model


def preprocess_image(image: Union[str, np.ndarray, Image.Image]) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Pre-process image for inference.
    
    Args:
        image: Image path (str), numpy array (H, W, 3), or PIL Image
    
    Returns:
        Tuple of (processed_tensor, original_size)
        - processed_tensor: [1, 3, 300, 300]
        - original_size: (height, width) of original image
    """
    # Load image
    if isinstance(image, str):
        pil_image = Image.open(image).convert('RGB')
    elif isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image).convert('RGB')
    elif isinstance(image, Image.Image):
        pil_image = image.convert('RGB')
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
    
    original_size = pil_image.size[::-1]  # (width, height) -> (height, width)
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    tensor = transform(pil_image)
    tensor = tensor.unsqueeze(0)  # Add batch dimension
    
    return tensor, original_size


def rescale_boxes(
    boxes: np.ndarray,
    from_size: Tuple[int, int],
    to_size: Tuple[int, int]
) -> np.ndarray:
    """
    Rescale bounding boxes from one image size to another.
    
    Args:
        boxes: Array of shape [N, 4] in [x1, y1, x2, y2] format
        from_size: Original image size (height, width)
        to_size: Target image size (height, width)
    
    Returns:
        Rescaled boxes
    """
    from_h, from_w = from_size
    to_h, to_w = to_size
    
    scale_x = to_w / from_w
    scale_y = to_h / from_h
    
    boxes = boxes.copy()
    boxes[:, [0, 2]] *= scale_x  # x1, x2
    boxes[:, [1, 3]] *= scale_y  # y1, y2
    
    return boxes


@torch.no_grad()
def run_inference(
    model: nn.Module,
    image: Union[str, np.ndarray, Image.Image],
    idx_to_class: Dict[int, str],
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    device: str = 'cuda',
) -> Dict:
    """
    Run inference on a single image.
    
    **CRITICAL:** Image tensors and model are explicitly moved to device.
    
    Args:
        model: Loaded SSD300 model
        image: Image path, numpy array, or PIL Image
        idx_to_class: Dictionary mapping class indices to class names
        confidence_threshold: Only keep detections above this score
        device: 'cuda' or 'cpu'
    
    Returns:
        Dictionary with keys:
        {
            'detections': [
                {
                    'bbox': [x1, y1, x2, y2],  # in original image coordinates
                    'label': 'cat',            # class name
                    'confidence': 0.95,
                },
                ...
            ],
            'image_size': [height, width],
            'num_detections': int
        }
    """
    # Pre-process image
    processed_image, original_size = preprocess_image(image)
    processed_image = processed_image.to(device)  # **CRITICAL: Move to device**
    
    # Run inference (wrapped in no_grad for efficiency)
    model.eval()
    predictions = model(processed_image)
    
    # Extract results from first (and only) image in batch
    pred = predictions[0]
    boxes = pred['boxes'].cpu().numpy()  # [N, 4] in [0, 1] normalized format
    scores = pred['scores'].cpu().numpy()  # [N]
    labels = pred['labels'].cpu().numpy()  # [N]
    
    # Rescale boxes back to original image size
    model_input_size = (300, 300)
    boxes = rescale_boxes(boxes, model_input_size, original_size)
    
    # Filter by confidence threshold
    mask = scores >= confidence_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    print ('lable from model:',labels)

    
    # Format results
    detections = []
    for bbox, score, label_idx in zip(boxes, scores, labels):
        # Clamp bbox coordinates to image boundaries
        h, w = original_size
        bbox[0] = np.clip(bbox[0], 0, w - 1)
        bbox[1] = np.clip(bbox[1], 0, h - 1)
        bbox[2] = np.clip(bbox[2], 0, w)
        bbox[3] = np.clip(bbox[3], 0, h)
        
        # Skip background class (index 0)
        if label_idx == 0:
            continue
        
        # Get class name from mapping
        label_name = idx_to_class.get(label_idx, f'unknown_{label_idx}')
        
        detections.append({
            'bbox': bbox.tolist(),  # [x1, y1, x2, y2]
            'label': label_name,
            'confidence': float(score),
        })
    
    # Sort by confidence (descending)
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    result = {
        'detections': detections,
        'image_size': list(original_size),
        'num_detections': len(detections),
    }
    
    return result


def batch_inference(
    model: nn.Module,
    images: List[Union[str, np.ndarray, Image.Image]],
    idx_to_class: Dict[int, str],
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    device: str = 'cuda',
) -> List[Dict]:
    """
    Run inference on multiple images.
    
    Args:
        model: Loaded SSD300 model
        images: List of images (paths or arrays)
        idx_to_class: Dictionary mapping class indices to class names
        confidence_threshold: Only keep detections above this score
        device: 'cuda' or 'cpu'
    
    Returns:
        List of detection results (one dict per image)
    """
    results = []
    for img in images:
        result = run_inference(model, img, idx_to_class, confidence_threshold, device)
        results.append(result)
    return results


# ============================================================================
# ======================== VISUALIZATION ====================================
# ============================================================================


def draw_detections_opencv(
    image: Union[str, np.ndarray],
    detections: Dict,
    output_path: Optional[str] = None,
    font_scale: float = 0.6,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw bounding boxes and labels on image using OpenCV.
    Useful for offline debugging and visualization.
    
    Args:
        image: Image path or numpy array
        detections: Output from run_inference()
        output_path: Optional path to save annotated image
        font_scale: Font size scale for labels
        thickness: Line thickness for bboxes
    
    Returns:
        Image with annotations (numpy array)
    """
    # Load image
    if isinstance(image, str):
        img_cv = cv2.imread(image)
        if img_cv is None:
            raise FileNotFoundError(f"Image not found: {image}")
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    else:
        img_cv = image.copy() if isinstance(image, np.ndarray) else np.array(image)
    
    # Draw each detection
    for det in detections['detections']:
        x1, y1, x2, y2 = [int(c) for c in det['bbox']]
        label = det['label']
        confidence = det['confidence']
        
        # Draw bounding box
        color = (0, 255, 0)  # Green
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label with confidence
        text = f"{label}: {confidence:.2f}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = x1
        text_y = max(y1 - 5, text_size[1])
        
        # Draw background for text
        cv2.rectangle(
            img_cv,
            (text_x, text_y - text_size[1] - 5),
            (text_x + text_size[0], text_y),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            img_cv,
            text,
            (text_x, text_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),  # Black text
            thickness
        )
    
    # Save if output path provided
    if output_path is not None:
        output_img = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, output_img)
        print(f"Annotated image saved to: {output_path}")
    
    return img_cv


def save_detections_json(
    detections: Union[Dict, List[Dict]],
    output_path: str,
    indent: int = 2,
) -> None:
    """
    Save detection results to JSON file.
    
    Args:
        detections: Single detection dict or list of dicts
        output_path: Path to save JSON
        indent: JSON indentation level
    """
    with open(output_path, 'w') as f:
        json.dump(detections, f, indent=indent)
    print(f"Detection results saved to: {output_path}")


# ============================================================================
# ======================== MAIN / EXAMPLE ===================================
# ============================================================================


def main():
    """Example usage of inference."""
    
    print(f"Device: {DEVICE}")
    print(f"Model path: {MODEL_PATH}")
    
    # Load class names from COCO JSON
    print("\nLoading class names from COCO annotation file...")
    idx_to_class = load_class_names_from_coco(COCO_ANN_FILE)
    print(f"Classes loaded: {idx_to_class}")
    
    # Load model
    print("\nLoading model...")
    model = load_model(MODEL_PATH, num_classes=NUM_CLASSES, device=str(DEVICE).split(':')[0])
    print("Model loaded successfully")
    
    # Example 1: Single image inference
    print("\n" + "=" * 80)
    print("Example 1: Single Image Inference")
    print("=" * 80)
    
    sample_image_path = './sample_image.jpg'
    if os.path.exists(sample_image_path):
        result = run_inference(
            model,
            sample_image_path,
            idx_to_class=idx_to_class,
            device=str(DEVICE).split(':')[0]
        )
        print(f"\nDetections found: {result['num_detections']}")
        for i, det in enumerate(result['detections']):
            print(f"  {i+1}. {det['label'].upper()} (confidence: {det['confidence']:.3f})")
            print(f"     Bbox: {[round(x, 1) for x in det['bbox']]}")
        
        # Draw and save annotated image
        annotated_img = draw_detections_opencv(
            sample_image_path,
            result,
            output_path='./inference_output_annotated.jpg'
        )
        # Save detection results as JSON
        save_detections_json(result, './inference_output.json')
    else:
        print(f"Sample image not found at {sample_image_path}")
        print("To use this example, place an image at:", sample_image_path)
    
    # Example 2: Batch inference
    print("\n" + "=" * 80)
    print("Example 2: Batch Inference (if multiple images available)")
    print("=" * 80)
    
    image_dir = './sample_images'
    if os.path.exists(image_dir):
        image_files = [
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        
        if image_files:
            print(f"Found {len(image_files)} images. Running inference...")
            results = batch_inference(
                model,
                image_files,
                idx_to_class=idx_to_class,
                device=str(DEVICE).split(':')[0]
            )
            
            # Save results
            save_detections_json(results, './batch_inference_results.json')
            print(f"Batch inference complete. Results saved to batch_inference_results.json")
        else:
            print(f"No images found in {image_dir}")
    else:
        print(f"Sample images directory not found at {image_dir}")


if __name__ == '__main__':
    # Uncomment the line below to run example inference
    # main()
    
    # For production use, import this module and use:
    # model = load_model('./checkpoints/best_ssd.pth', device='cuda')
    # result = run_inference(model, 'path/to/image.jpg', device='cuda')
    
    print("Inference module loaded successfully.")
    print("To run inference programmatically, use:")
    print("  from inference import load_model, run_inference")
    print("  model = load_model('./checkpoints/best_ssd.pth', device='cuda')")
    print("  result = run_inference(model, 'image.jpg', device='cuda')")
