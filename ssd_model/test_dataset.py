"""
Test script to verify dataset loading and COCO JSON parsing.
Run this before training to ensure everything is set up correctly.
"""

import sys
import torch
from dataset import AnimalDataset, get_ssd_transforms


def test_dataset_loading():
    """Test that dataset loads correctly from COCO JSON."""
    
    print("=" * 80)
    print("COCO Dataset Loading Test")
    print("=" * 80)
    
    # Configuration (update these paths if needed)
    IMG_DIR = './animal_dataset/train'
    ANN_FILE = './animal_dataset/train/_annotations.coco.json'
    
    print(f"\nDataset paths:")
    print(f"  Images: {IMG_DIR}")
    print(f"  Annotations: {ANN_FILE}")
    
    # Try loading the dataset
    try:
        print("\n[1/3] Loading dataset...")
        dataset = AnimalDataset(
            img_dir=IMG_DIR,
            ann_file=ANN_FILE,
            annotation_format='coco',
            transform=get_ssd_transforms(is_train=True),
        )
        print("✓ Dataset loaded successfully!")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("\nPlease ensure:")
        print(f"  1. Folder '{IMG_DIR}' exists with training images")
        print(f"  2. File '{ANN_FILE}' exists with COCO JSON annotations")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test dataset properties
    print("\n[2/3] Dataset properties:")
    num_samples = len(dataset)
    num_classes = dataset.get_num_classes()
    print(f"  Total samples: {num_samples}")
    print(f"  Total classes (including background): {num_classes}")
    
    # Print class mapping
    print(f"  Class mapping:")
    for idx in range(1, num_classes):  # Skip background (index 0)
        class_name = dataset.get_class_name(idx)
        print(f"    {idx}: {class_name}")
    
    if num_samples == 0:
        print("✗ Error: Dataset is empty. Check annotation file.")
        return False
    
    # Test loading a sample
    print("\n[3/3] Loading sample...")
    try:
        image, target = dataset[0]
        print("✓ Sample loaded successfully!")
        print(f"  Image shape: {image.shape} (expected: [3, 300, 300])")
        print(f"  Number of objects: {len(target['labels'])}")
        print(f"  Boxes shape: {target['boxes'].shape}")
        print(f"  Labels shape: {target['labels'].shape}")
        
        # Verify types
        if not isinstance(image, torch.Tensor):
            print(f"✗ Error: Image is {type(image)}, expected torch.Tensor")
            return False
        
        if image.dtype != torch.float32:
            print(f"✗ Error: Image dtype is {image.dtype}, expected torch.float32")
            return False
        
        if target['boxes'].dtype != torch.float32:
            print(f"✗ Error: Boxes dtype is {target['boxes'].dtype}, expected torch.float32")
            return False
        
        if target['labels'].dtype != torch.int64:
            print(f"✗ Error: Labels dtype is {target['labels'].dtype}, expected torch.int64")
            return False
        
        print("✓ All data types are correct!")
        
        # Print first sample details
        if len(target['labels']) > 0:
            print(f"\n  First object in sample:")
            label_idx = target['labels'][0].item()
            label_name = dataset.get_class_name(label_idx)
            bbox = target['boxes'][0].numpy()
            print(f"    Label: {label_name} (index {label_idx})")
            print(f"    Bounding box (Pascal VOC format): {bbox}")
            print(f"    Box format check: x_min={bbox[0]:.1f}, y_min={bbox[1]:.1f}, x_max={bbox[2]:.1f}, y_max={bbox[3]:.1f}")
        
    except Exception as e:
        print(f"✗ Error loading sample: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test GPU availability
    print("\n[BONUS] GPU Status:")
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Try moving tensor to GPU
        try:
            image_gpu = image.to('cuda')
            print(f"✓ Successfully moved image to GPU")
        except Exception as e:
            print(f"✗ Error moving tensor to GPU: {e}")
    else:
        print("⚠ CUDA is not available. Training will use CPU (much slower).")
    
    print("\n" + "=" * 80)
    print("✓ All tests passed! Dataset is ready for training.")
    print("=" * 80)
    return True


if __name__ == '__main__':
    success = test_dataset_loading()
    sys.exit(0 if success else 1)
