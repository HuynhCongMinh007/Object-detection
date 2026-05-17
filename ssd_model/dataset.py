"""
Custom Dataset class for SSD-based object detection.
Designed for COCO JSON format with proper bbox conversion.
Avoids pycocotools to prevent Windows C++ compiler issues.
"""

import os
import json
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class AnimalDataset(Dataset):
    """
    Custom PyTorch Dataset for animal detection using COCO JSON format.
    
    **CRITICAL:** COCO stores bboxes as [x_min, y_min, width, height].
    This dataset automatically converts to Pascal VOC format: [x_min, y_min, x_max, y_max]
    which is required by PyTorch's SSD model.
    
    Args:
        img_dir (str): Path to directory containing images
        ann_file (str): Path to COCO JSON annotation file (_annotations.coco.json)
        annotation_format (str): Either 'coco' (default) or 'voc'
        transform (callable, optional): Image transformations to apply
    """
    
    def __init__(
        self,
        img_dir: str,
        ann_file: str,
        annotation_format: str = 'coco',
        transform: Optional[callable] = None,
    ):
        self.img_dir = img_dir
        self.ann_file = ann_file
        self.annotation_format = annotation_format.lower()
        self.transform = transform
        
        # Load COCO annotations and build internal mappings
        self.images_list = []
        self.annotations = {}  # image_filename -> {'boxes': array, 'labels': array}
        self.class_name_to_idx = {}  # class_name -> index
        self.idx_to_class_name = {}  # index -> class_name
        
        self._load_coco_annotations()
    
    def _load_coco_annotations(self):
        """
        Load COCO JSON annotations.
        
        Automatically converts COCO bbox format [x, y, w, h] to Pascal VOC format [x, y, x+w, y+h].
        """
        if not os.path.exists(self.ann_file):
            raise FileNotFoundError(f"Annotation file not found: {self.ann_file}")
        
        with open(self.ann_file, 'r') as f:
            coco_data = json.load(f)
        
        print(f"[AnimalDataset] Loaded COCO JSON from: {self.ann_file}")
        
        # Build category index mapping: category_id -> category_name
        # Also keep a mapping category_name -> index (index 0 is reserved for background)
        category_map = {}  # category_id -> category_name
        for cat in coco_data.get('categories', []):
            cat_id = cat['id']
            cat_name = cat['name']
            category_map[cat_id] = cat_name
            # Store mapping for later retrieval (1-indexed for actual classes)
            if cat_name not in self.class_name_to_idx:
                class_idx = len(self.class_name_to_idx) + 1  # 1-indexed (0 is background)
                self.class_name_to_idx[cat_name] = class_idx
                self.idx_to_class_name[class_idx] = cat_name
        
        print(f"[AnimalDataset] Found {len(self.class_name_to_idx)} animal classes:")
        for name, idx in sorted(self.class_name_to_idx.items(), key=lambda x: x[1]):
            print(f"  {idx}: {name}")
        
        # Build mapping: image_id -> image info
        images_map = {img['id']: img for img in coco_data['images']}
        
        # Group annotations by image_id
        annotations_by_image = {}
        for ann in coco_data.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
        
        # Build images list and annotations dict
        for img_id, img_info in images_map.items():
            img_filename = img_info['file_name']
            self.images_list.append(img_filename)
            
            # Extract bounding boxes and labels
            boxes = []
            labels = []
            for ann in annotations_by_image.get(img_id, []):
                bbox = ann['bbox']  # COCO format: [x, y, width, height]
                
                # **CRITICAL CONVERSION:** Convert COCO [x, y, w, h] to Pascal VOC [x_min, y_min, x_max, y_max]
                x_min, y_min, width, height = bbox
                x_max = x_min + width
                y_max = y_min + height
                boxes.append([x_min, y_min, x_max, y_max])
                
                # Get class index (1-indexed; 0 is reserved for background)
                cat_id = ann['category_id']
                cat_name = category_map.get(cat_id, 'unknown')
                class_idx = self.class_name_to_idx.get(cat_name, 1)  # Default to class 1 if unknown
                labels.append(class_idx)
            
            # Store as float32 for boxes, int64 for labels (PyTorch requirements)
            self.annotations[img_filename] = {
                'boxes': np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32),
                'labels': np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64),
            }
        
        print(f"[AnimalDataset] Loaded {len(self.images_list)} images")
    
    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.images_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Get a sample and its annotations.
        
        Returns:
            image (torch.Tensor): Image tensor of shape [3, H, W]
            target (Dict): Dictionary with keys:
                - 'boxes': torch.Tensor of shape [N, 4] in [x_min, y_min, x_max, y_max] format (Pascal VOC)
                - 'labels': torch.Tensor of shape [N] with 1-indexed class indices (0 is background, not used)
        """
        img_filename = self.images_list[idx]
        img_path = os.path.join(self.img_dir, img_filename)
        
        # Load image
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        image = Image.open(img_path).convert('RGB')

        # Get annotations (numpy arrays)
        ann = self.annotations[img_filename]
        boxes_np = ann['boxes'].copy()  # Shape: [N, 4] in original image coordinates
        labels = torch.from_numpy(ann['labels'])  # Shape: [N]

        # Apply transformations to image AND adjust boxes to match resized image.
        # Our transforms currently perform a fixed resize to 300x300 (no geometric random
        # flips) so we can safely scale box coordinates.
        orig_w, orig_h = image.size  # PIL: (width, height)
        target_w, target_h = (300, 300)

        # Scale boxes from original image size to model input size
        if boxes_np.size > 0:
            scale_x = float(target_w) / float(orig_w)
            scale_y = float(target_h) / float(orig_h)
            boxes_np[:, 0] = boxes_np[:, 0] * scale_x  # x_min
            boxes_np[:, 1] = boxes_np[:, 1] * scale_y  # y_min
            boxes_np[:, 2] = boxes_np[:, 2] * scale_x  # x_max
            boxes_np[:, 3] = boxes_np[:, 3] * scale_y  # y_max

        boxes = torch.from_numpy(boxes_np)

        # Apply image-only transforms (resize, to tensor, normalize)
        if self.transform is not None:
            image = self.transform(image)
        
        # Return standard PyTorch detection format
        target = {
            'boxes': boxes,      # Float tensor [N, 4] in Pascal VOC format (scaled to 300x300)
            'labels': labels,    # Int64 tensor [N] with 1-indexed class indices
        }
        
        return image, target
    
    def get_class_name(self, class_idx: int) -> str:
        """
        Get class name from 1-indexed class index.
        
        Args:
            class_idx: 1-indexed class index (0 is background, not used)
        
        Returns:
            Class name string
        """
        return self.idx_to_class_name.get(class_idx, 'unknown')
    
    def get_num_classes(self) -> int:
        """
        Return total number of classes including background.
        
        Returns:
            num_classes = len(animal_classes) + 1 (for background at index 0)
        """
        return len(self.class_name_to_idx) + 1


# Default transformations optimized for SSD300
# Default transformations optimized for SSD300
def get_ssd_transforms(is_train: bool = True) -> transforms.Compose:
    """
    Get image transformations for SSD300.
    SSD300 standard input size: 300x300
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            # Normalize using ImageNet statistics
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])


# Example usage
if __name__ == '__main__':
    # Example: COCO format
    # dataset = AnimalDataset(
    #     img_dir='./animal_dataset/train',
    #     ann_file='./animal_dataset/train/_annotations.coco.json',
    #     annotation_format='coco',
    #     transform=get_ssd_transforms(is_train=True),
    # )
    # print(f"Dataset size: {len(dataset)}")
    # print(f"Num classes (with background): {dataset.get_num_classes()}")
    # sample_image, sample_target = dataset[0]
    # print(f"Image shape: {sample_image.shape}")
    # print(f"Number of objects: {len(sample_target['labels'])}")
    # for i, label_idx in enumerate(sample_target['labels']):
    #     print(f"  Object {i}: {dataset.get_class_name(label_idx.item())}")
    pass
