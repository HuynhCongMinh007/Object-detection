# 🚀 YOLOv8 Nano - Animal Object Detection

**Ultra-lightweight, real-time object detection model optimized for edge devices and rapid inference.**

> **Model:** YOLOv8 Nano (6.3 MB)  
> **Framework:** PyTorch + Ultralytics  
> **Classes:** 6 animal types  
> **Status:** ✅ Ready for Production  

---

## 📋 Overview

YOLOv8 Nano is the smallest variant of YOLO version 8, offering:
- ⚡ **Ultra-fast inference** (50-100 FPS on CPU, 300+ FPS on GPU)
- 📦 **Minimal model size** (6.3 MB - deployable on mobile/edge devices)
- 🎯 **Real-time detection** with competitive accuracy
- 🐍 **Easy-to-use Python API** from Ultralytics

Perfect for:
- Edge device deployment (Raspberry Pi, Jetson Nano)
- Real-time streaming applications
- Mobile applications
- Resource-constrained environments

---

## 🛠️ Installation

### Requirements
- Python 3.8 or higher
- CUDA 11.8+ (optional, for GPU acceleration)

### Setup

```bash
# Navigate to YOLOv8 model directory
cd Yolov8Nano_model

# Install dependencies
pip install -r requirements.txt
```

**Required packages:**
```
ultralytics>=8.0.0
opencv-python>=4.6.0
```

---

## � Dataset Download

The complete dataset with images and annotations can be downloaded from:

**🔗 [Download Dataset (Google Drive)](https://drive.google.com/file/d/1eF_6vKJ7k5Mxi8HEosJBbd08svld3SLA/view?usp=sharing)**

**Setup Instructions:**
1. Download the dataset from the link above
2. Extract the compressed file
3. Place the `data/` folder inside the `Yolov8Nano_model/` directory
4. Verify the directory structure matches the layout below

---

## �📊 Dataset Structure

```
data/
├── data.yaml                    # Dataset configuration
├── train/
│   ├── images/                  # Training images (JPEG/PNG)
│   └── labels/                  # YOLO format annotations (.txt)
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### YOLO Label Format
Each image has a corresponding `.txt` file with detections:
```
<class_id> <x_center> <y_center> <width> <height>
<class_id> <x_center> <y_center> <width> <height>
...
```

**Example:** `cat_1.txt`
```
0 0.5 0.5 0.3 0.4
```
(Class 0=cat, center at 50%x50%, 30% width, 40% height)

### Classes (data.yaml)
```yaml
nc: 6
names: ['cat', 'chicken', 'cow', 'dog', 'horse', 'sheep']
```

---

## 🎓 Training

### Basic Training Command

```bash

# Full training (50 epochs)
python train.py --epochs 50

# Custom batch size
python train.py --epochs 50 --batch-size 16
```

### Training Configuration

Edit in `train.py`:
```python
DATA_YAML = './data/data.yaml'          # Dataset config file
MODEL_TYPE = 'yolov8n.pt'               # nano/small/medium/large
IMG_SIZE = 640                          # Input resolution
DEVICE = 'cpu'                          # 'cpu' or '0' for GPU
BATCH_SIZE = 16                         # Adjust for your hardware
PROJECT_NAME = 'runs/train'             # Output directory
```

### Training Output

Best model is automatically saved to:
```
checkpoints/best.pt
```

Training logs and visualizations:
```
runs/train/yolov8_animals/
├── weights/
│   ├── best.pt                 # Best model
│   └── last.pt                 # Last checkpoint
├── results.png                 # Training curves
└── confusion_matrix.png        # Confusion matrix
```

---

## 🔍 Inference & Validation

### Inference Commands

**Single Image:**
```bash
python inference.py --source image.jpg --conf 0.5
```

**Batch Folder (with statistics):**
```bash
python inference.py --source ./data/test/images/ --conf 0.5
```

**Custom Output Directory:**
```bash
python inference.py --source ./data/test/images/ --conf 0.5 --output ./results
```

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--source` | str | `test.jpg` | Path to image or folder |
| `--conf` | float | `0.5` | Confidence threshold (0.0-1.0) |
| `--output` | str | `./outputs` | Output directory for results |

### Validation Command

**Evaluate on Test Set:**
```bash
python val.py
```

Generates:
- ✅ mAP@0.5 and mAP@0.5:0.95 scores
- ✅ Per-class Precision & Recall
- ✅ Confusion matrix
- ✅ PR curves and loss plots
- ✅ Sample predictions (val_batch*.jpg)

### Output Format

**Batch Results JSON** (`outputs/batch_results.json`):
```json
{
  "cat_1.jpg": {
    "detections": [
      {
        "bbox": [100, 150, 250, 300],
        "label": "cat",
        "confidence": 0.95
      },
      {
        "bbox": [350, 200, 500, 400],
        "label": "dog",
        "confidence": 0.88
      }
    ],
    "image_size": [640, 480],
    "num_detections": 2
  },
  "cat_2.jpg": {
    "detections": [],
    "image_size": [640, 480],
    "num_detections": 0
  }
}
```

### Batch Inference

Process entire folders of images with visualization:
```bash
python inference.py --source ./data/test/images/ --conf 0.5
```

**Console Output Example:**
```
======================================================================
📊 TÓM TẮT KẾT QUẢ INFERENCE
======================================================================

📷 TỔNG QUAN:
   • Tổng số ảnh: 172
   • Ảnh có phát hiện: 145 (84.3%)
   • Ảnh không có phát hiện: 27
   • Tổng số detections: 268
   • Confidence threshold: 0.5

🐾 THỐNG KÊ THEO CLASS:
Class           Count      Avg Conf     Min        Max       
----------------------------------------------------------------------
cat             85         0.9234       0.5012     0.9912
chicken         28         0.8756       0.5234     0.9567
cow             32         0.9012       0.5678     0.9854
dog             43         0.8145       0.5123     0.9456
horse           32         0.8567       0.5345     0.9678
sheep           44         0.7890       0.5012     0.9234
======================================================================
```

### Output Structure

Predictions are saved with bounding boxes and detailed statistics:
```
outputs/
├── predictions/                # Images with bounding boxes & labels
│   ├── image_1_pred.jpg
│   ├── image_2_pred.jpg
│   └── ...                     # One image per input file
└── batch_results.json          # Complete detection results (JSON)
```

### Validation

Evaluate model on test set:
```bash
python val.py
```

**Output includes:**
- mAP (mean Average Precision) scores per class
- Precision & Recall metrics
- Confusion matrix visualization
- PR curves
- Results saved to: `runs/detect/val/`

---

## ⚙️ Hyperparameters

Adjust these in `train.py`:

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `epochs` | 50 | 1-500 | Number of training iterations |
| `batch_size` | 16 | 1-128 | Images per batch (higher = more VRAM) |
| `imgsz` | 640 | 320-1280 | Input image resolution |
| `device` | 'cpu' | 'cpu', '0', '1' | Training device |
| `optimizer` | 'auto' | 'SGD', 'Adam' | Optimizer type |

---

### Accuracy (Test Set)

Expected performance after 50 epochs:
- **mAP@0.5:** 84.5%
- **mAP@0.5:0.95** 57.3%
- **Precision:** 88.7%
- **Recall:** 74.6%

*Results depend on dataset quality and augmentation*

---

## 🔗 Useful Commands

```bash
# Validate model
python inference.py --source data/valid/images/ --conf 0.5

# Export model to ONNX (cross-platform)
python -c "from ultralytics import YOLO; m = YOLO('checkpoints/best.pt'); m.export(format='onnx')"

# Export to TensorFlow Lite (mobile)
python -c "from ultralytics import YOLO; m = YOLO('checkpoints/best.pt'); m.export(format='tflite')"
```

---

## 📊 Directory Structure

```
Yolov8Nano_model/
├── train.py                    # Training script
├── inference.py                # Inference with batch & visualization (UPDATED)
├── val.py                      # Validation script (NEW)
├── requirements.txt
├── yolov8n.pt                  # Pre-trained weights
├── checkpoints/
│   ├── best.pt                 # Best performing model ⭐
│   ├── best2.pt
│   └── best3.pt
├── runs/
│   └── detect/
│       ├── train/              # Training logs & curves
│       │   └── yolov8_animals/
│       │       ├── weights/
│       │       ├── results.png
│       │       └── confusion_matrix.png
│       └── val/                # Validation metrics (NEW)
│           ├── confusion_matrix.png
│           ├── BoxP_curve.png
│           ├── BoxR_curve.png
│           ├── BoxF1_curve.png
│           ├── val_batch0_pred.jpg
│           └── ...
├── outputs/
│   ├── predictions/            # Images with bounding boxes
│   │   ├── image_1_pred.jpg
│   │   ├── image_2_pred.jpg
│   │   └── ...
│   └── batch_results.json      # Detection results (NEW)
└── data/
    ├── data.yaml
    ├── train/
    ├── valid/
    └── test/
```

---

## 📖 References

- [YOLOv8 Official Docs](https://docs.ultralytics.com/models/yolov8/)
- [YOLO Paper](https://arxiv.org/abs/2307.02788)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
- [PyTorch Documentation](https://pytorch.org/)

---

