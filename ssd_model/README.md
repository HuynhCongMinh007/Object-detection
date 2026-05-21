# SSD Model - Animal Object Detection

**Status:** Production-ready SSD300-VGG16 model for 6-class animal detection (5 animals + 1 background class)

## Overview

This module implements an SSD300-VGG16 (Single Shot MultiBox Detector) model for real-time object detection of animals in images. It uses **transfer learning** with pre-trained COCO weights from PyTorch's `torchvision`, making it efficient and accurate even with limited training data.

### Key Features

- ✅ Pre-trained model from torchvision (COCO dataset)
- ✅ Custom classification head for 6 classes (5 animals + background)
- ✅ Support for both COCO JSON and Pascal VOC XML annotation formats
- ✅ Optimized batch sizes for RTX 3050 GPU (8-16 samples per batch)
- ✅ Clean inference API with JSON output
- ✅ Visualization utilities for debugging
- ✅ Mixed precision training support (easily extensible)

## Project Structure

```
📂 ssd_model/
├── requirements.txt          # Python dependencies
├── dataset.py               # Custom AnimalDataset class
├── train.py                 # Training script with hyperparameters
├── inference.py             # Inference and visualization utilities
├── checkpoints/             # (Created during training) Saves best_ssd.pth
├── README.md                # This file
└── data/                    # (Structure should match annotation format)
    ├── train/
    │   ├── images/          # Training images (JPEG, PNG)
    │   └── annotations.[json|] / annotations/ (for VOC)
    └── val/
        ├── images/          # Validation images
        └── annotations.[json|] / annotations/ (for VOC)
```

## System Requirements

- **GPU:** NVIDIA GPU with CUDA support (tested on RTX 3050)
- **CPU:** Multi-core processor (for data loading)
- **RAM:** 32GB (sufficient for batch size 8-16)
- **Storage:** ~5GB for model checkpoints + dataset
- **Python:** 3.8+

## Installation

### Step 1: Clone/Navigate to the Repository

```bash
cd /path/to/Object-detection/ssd_model
```

### Step 2: Create Python Virtual Environment

```bash
# On Windows
python -m venv .venv
.venv\Scripts\activate

# On macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**Expected output:** All packages installed without errors. Verify with:

```bash
pip list
```

## Data Preparation

### Annotation Format

The dataset supports **two annotation formats**:

#### Option A: COCO JSON Format

```json
{
  "images": [
    {"id": 1, "file_name": "image001.jpg", "width": 640, "height": 480}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 2,
      "bbox": [x, y, width, height]
    }
  ],
  "categories": [
    {"id": 0, "name": "background"},
    {"id": 1, "name": "cat"},
    {"id": 2, "name": "dog"}
  ]
}
```

**File structure:**

```
data/train/
├── images/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
└── annotations.json
```

#### Option B: Pascal VOC XML Format

```xml
<?xml version="1.0"?>
<annotation>
  <filename>image001.jpg</filename>
  <object>
    <name>cat</name>
    <bndbox>
      <xmin>100</xmin>
      <ymin>50</ymin>
      <xmax>300</xmax>
      <ymax>400</ymax>
    </bndbox>
  </object>
</annotation>
```

**File structure:**

```
data/train/
├── images/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
└── annotations/
    ├── image001.xml
    ├── image002.xml
    └── ...
```

### Class Names

Update `CLASS_NAMES` in both `train.py` and `inference.py`:

```python
CLASS_NAMES = [
    'background',  # Index 0 (always required)
    'cat',         # Index 1
    'dog',         # Index 2
    'bird',        # Index 3
    'fox',         # Index 4
    'rabbit',      # Index 5
]
```

## Training

### Step 1: Update Paths in `train.py`

Edit the configuration section at the top of `train.py`:

```python
# Paths (modify these to point to your dataset)
TRAIN_IMG_DIR = './data/train/images'
TRAIN_ANN_FILE = './data/train/annotations.json'  # or './data/train/annotations' for VOC
VAL_IMG_DIR = './data/val/images'
VAL_ANN_FILE = './data/val/annotations.json'
ANNOTATION_FORMAT = 'coco'  # 'coco' or 'voc'
```

### Step 2: Tune Hyperparameters (Optional)

```python
EPOCHS = 100              # Number of training epochs
BATCH_SIZE = 8            # Start with 8, increase to 16 if memory allows
LEARNING_RATE = 0.001     # SGD learning rate
WEIGHT_DECAY = 5e-4       # L2 regularization
MOMENTUM = 0.9            # SGD momentum
```

### Step 3: Run Training

```bash
python train.py --epochs 100 --batch-size 8 --lr 0.001
```

**During training, you'll see:**

```
Device: cuda
Num Classes: 6
Batch Size: 8

[1/4] Loading datasets...
Training samples: 500
Validation samples: 100

[2/4] Creating data loaders...
[3/4] Loading pre-trained SSD300-VGG16 model...
[4/4] Setting up optimizer and scheduler...

================================================================================
Starting training...
================================================================================

Epoch [1/100]
  Epoch [1] Batch [50/63] Loss: 2.1234 (Avg: 2.4567)
  Train Loss: 2.3456
  Val Loss: 2.0123
  ✓ Best model saved to ./checkpoints/best_ssd.pth

Epoch [2/100]
  ...
```

### Step 4: Monitor Training

The best model (lowest validation loss) is automatically saved to:

```
./checkpoints/best_ssd.pth
```

**Tips:**

- If CUDA runs out of memory, reduce `BATCH_SIZE` to 4
- If training is too slow, increase `num_workers` in the DataLoader
- Early stopping can be implemented by checking validation loss plateau

## Inference

### Step 1: Prepare Inference Script

Edit `inference.py` to point to your best model and set class names:

```python
MODEL_PATH = './checkpoints/best_ssd.pth'
NUM_CLASSES = 6
CLASS_NAMES = ['background', 'cat', 'dog', 'bird', 'fox', 'rabbit']
CONFIDENCE_THRESHOLD = 0.5
```

### Step 2: Run Inference on Single Image

Sử dụng `inference_runner.py` để chạy dự đoán trên một ảnh bất kỳ. File này cho phép truyền đường dẫn ảnh và model checkpoint thông qua command line arguments.

```bash
python inference_runner.py -i <path_to_image.jpg> -o <path_to_output_image.jpg> -m ./checkpoints/best_ssd.pth
```

**Ví dụ:**

```bash
python inference_runner.py -i ./animal_dataset/test/cat.jpg -o ./cat_result.jpg
```

Hoặc sử dụng trong code Python với `inference.py`:

```python
from inference import load_model, run_inference

# Load model
model = load_model('./checkpoints/best_ssd.pth', device='cuda')

# Run inference
result = run_inference(model, 'path/to/image.jpg', device='cuda')

# Print results
print(f"Detections: {result['num_detections']}")
for det in result['detections']:
    print(f"  - {det['label']} ({det['confidence']:.2f})")
    print(f"    Bbox: {det['bbox']}")
```

### Step 3: Đánh giá mô hình (Evaluation)

Để đánh giá toàn diện mô hình trên tập test, bao gồm các chỉ số mAP, Precision, Recall, F1-score và tạo các biểu đồ (Loss Curve, Confusion Matrix), sử dụng `evaluate_metrics.py`.

```bash
python evaluate_metrics.py
```

Sau khi chạy xong, các kết quả sẽ được lưu trong thư mục `outputs/`:

- `outputs/evaluation_report.md`: Báo cáo chi tiết các độ đo.
- `outputs/loss_curve.png`: Biểu đồ hàm loss trong quá trình huấn luyện.
- `outputs/confusion_matrix.png`: Biểu đồ ma trận nhầm lẫn dự đoán và thực tế.

### Step 4: Output Format

Inference returns a clean dictionary:

```python
{
    'detections': [
        {
            'bbox': [100, 50, 300, 400],     # [x1, y1, x2, y2] in original image coords
            'label': 'cat',                   # class name
            'confidence': 0.95,               # probability score
        },
        {
            'bbox': [150, 200, 280, 350],
            'label': 'dog',
            'confidence': 0.87,
        }
    ],
    'image_size': [480, 640],       # [height, width] of image
    'num_detections': 2
}
```

### Step 4: Batch Inference

```python
from inference import batch_inference, save_detections_json

results = batch_inference(
    model,
    ['image1.jpg', 'image2.jpg', 'image3.jpg'],
    device='cuda'
)

# Save results to JSON
save_detections_json(results, './batch_results.json')
```

### Step 5: Visualize Predictions

```python
from inference import draw_detections_opencv

annotated_img = draw_detections_opencv(
    'path/to/image.jpg',
    result,
    output_path='output_with_boxes.jpg'
)
```

## File Descriptions

### `requirements.txt`

Minimal dependencies for training and inference:

- `torch`: PyTorch deep learning framework
- `torchvision`: Pre-trained models and utilities
- `opencv-python`: Image I/O and visualization
- `pillow`: Image processing
- `numpy`: Numerical operations

### `dataset.py`

**Key Classes:**

- `AnimalDataset`: Custom PyTorch Dataset class
  - Loads images and annotations from COCO JSON or Pascal VOC XML
  - Applies image transformations (resize to 300x300, normalize)
  - Returns image tensor + target dict with bboxes and labels

- `get_ssd_transforms()`: Returns standard SSD300 transformations
  - Training: includes random horizontal flip
  - Validation: deterministic resize and normalization

**Usage:**

```python
dataset = AnimalDataset(
    img_dir='./data/train/images',
    ann_file='./data/train/annotations.json',
    class_names=CLASS_NAMES,
    annotation_format='coco',
    transform=get_ssd_transforms(is_train=True),
)
```

### `train.py`

**Key Functions:**

- `train_one_epoch()`: Trains for one epoch, returns avg loss
- `evaluate()`: Validates model, returns avg loss
- `modify_ssd_head()`: Replaces classification head for custom classes
- `load_model()`: Loads pre-trained SSD300 and modifies for our task
- `main()`: Complete training pipeline

**Hyperparameters** (easily tunable at top of file):

```python
EPOCHS = 100
BATCH_SIZE = 8
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4
```

**Output:**

- Saves best model to `./checkpoints/best_ssd.pth`
- Prints training/validation losses each epoch

### `inference.py`

**Key Functions:**

- `load_model()`: Loads trained .pth weights
- `run_inference()`: Runs inference on single image, returns dict
- `batch_inference()`: Runs inference on multiple images
- `draw_detections_opencv()`: Visualizes predictions on image
- `save_detections_json()`: Saves results to JSON file

**Output:**

- Returns clean dictionary with detections (bboxes, labels, scores)
- Compatible with frontend/backend teams (JSON serializable)

## Troubleshooting

### CUDA Out of Memory

**Solution:** Reduce `BATCH_SIZE` in `train.py`

```python
BATCH_SIZE = 4  # Instead of 8 or 16
```

### Model doesn't load

**Verify:**

1. Path exists: `./checkpoints/best_ssd.pth`
2. File is not corrupted: Check file size (should be ~100-200MB)
3. Number of classes matches (must be 6 for this config)

### Low detection accuracy

**Improvements:**

1. Increase training epochs (`EPOCHS = 200`)
2. Adjust learning rate (`LEARNING_RATE = 0.0001` for fine-tuning)
3. Ensure dataset has sufficient annotations per class (>100 per class recommended)
4. Check that class names match your data exactly

### Data loading is slow

**Solution:** Increase `num_workers` in DataLoader:

```python
# In train.py, increase from 0 to 4 or 8
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,  # Increase this
    collate_fn=collate_fn,
)
```

## Performance Expectations

On RTX 3050 with batch size 8:

- **Training speed:** ~30-50 ms per batch
- **Inference speed:** ~50-100 ms per image (300x300)
- **Model size:** ~100MB
- **VRAM usage:** ~6-8GB during training

## Integration with Other Modules

This SSD module outputs standardized JSON-compatible detection results:

```python
result = {
    'detections': [
        {'bbox': [x1, y1, x2, y2], 'label': 'cat', 'confidence': 0.95},
        ...
    ],
    'image_size': [h, w],
    'num_detections': n
}
```

Your **YOLO** and **Transformers** modules should follow the same output format for consistent backend integration.

## Team Collaboration

### Repository Structure

```
Object-detection/ (Git root)
├── ssd_model/          # Your module (this folder)
├── yolo_model/         # Team member's module
├── transformers_model/ # Team member's module
└── README.md
```

### Integration Points

1. **Dataset:** Ensure all modules use same class names and order
2. **Output format:** All modules should output the same detection dictionary
3. **Model checkpoints:** Store in separate `checkpoints/` folders per module

## References

- **PyTorch SSD:** https://pytorch.org/vision/stable/models/ssd.html
- **torchvision Transfer Learning:** https://pytorch.org/vision/stable/models.html
- **COCO Dataset:** https://cocodataset.org/
- **SSD Paper:** https://arxiv.org/abs/1512.02325

## Authors & License

Created for university final project (Object Detection Application for kids).

---

**Last Updated:** 2026-05-16  
**Status:** ✅ Production Ready
