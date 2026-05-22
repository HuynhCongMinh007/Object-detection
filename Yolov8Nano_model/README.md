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

## 🌀 Offline Data Augmentation

To programmatically augment the original dataset of 1,722 images (1,206 in training split) to a target of 4,000+ training images without relying on Roboflow's online augmentations, we use the custom script [dataset.py](file:///d:/Downloads/Object-detection/Yolov8Nano_model/dataset.py).

### Augmentation Methods
The script implements the following techniques using OpenCV and Numpy:
1. **Horizontal Flip** (automatically updates bounding boxes and polygon coordinate arrays)
2. **Brightness & Contrast Jittering** (random multipliers and additions)
3. **Gaussian Blur** (simulates defocusing)
4. **Gaussian Noise** (simulates camera sensor noise)
5. **Random Shift & Scale** (translations up to 6% and zoom factors between 0.85-1.15; automatically filters out objects pushed off-screen)

Supports both standard YOLO bounding boxes (`[xc, yc, w, h]`) and YOLO instance segmentation polygons (`[x1, y1, x2, y2, ...]`).

### CLI Usage
You can run the dataset augmentation script independently:
```bash
# Clean previous manual augmentations and generate up to 4,000 train images
python dataset.py --clean --target-count 4000
```
This writes augmented images labeled with the `_manual_aug_` suffix in `data/train/images/` and `data/train/labels/`.

---

## 🎓 Training

### Basic Training Command

> [!NOTE]
> [train.py](file:///d:/Downloads/Object-detection/Yolov8Nano_model/train.py) now automatically checks the number of training images. If it is less than 4,000, it automatically triggers [dataset.py](file:///d:/Downloads/Object-detection/Yolov8Nano_model/dataset.py) to perform offline augmentation before starting the training process.
> It also auto-detects CUDA availability, falling back to CPU if no GPU is found, to prevent device compatibility crashes.

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

### Inference Commands (Sử dụng [inference.py](file:///d:/Downloads/Object-detection/Yolov8Nano_model/inference.py))

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

### Validation Command (Sử dụng [val.py](file:///d:/Downloads/Object-detection/Yolov8Nano_model/val.py))

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
   • Ảnh có phát hiện: 158 (91.9%)
   • Ảnh không có phát hiện: 14
   • Tổng số detections: 223
   • Confidence threshold: 0.5

🐾 THỐNG KÊ THEO CLASS:
Class           Count      Avg Conf     Min        Max       
----------------------------------------------------------------------
cat             30         0.8153       0.6148     0.9595    
chicken         47         0.8256       0.5369     0.9309    
cow             54         0.7947       0.5064     0.9578    
dog             39         0.8089       0.5259     0.9413    
horse           28         0.8332       0.5530     0.9503    
sheep           25         0.8195       0.6045     0.9516    
======================================================================
```

> 📌 **Lưu ý về cột Count (Số lượng):** Cột `Count` thể hiện tổng số **cá thể/đối tượng** được mô hình phát hiện ra (không phải số lượng tệp ảnh). Vì một ảnh có thể chứa nhiều cá thể (ví dụ: một đàn gà gồm nhiều con, hoặc cả chó lẫn mèo), nên tổng số detections phát hiện được (223 detections) sẽ lớn hơn tổng số lượng tệp ảnh của tập test (172 ảnh).

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

Actual performance measured on the test set (`data/test`) using the checkpoint trained on our custom augmented dataset:

| Class | Precision (P) | Recall (R) | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| **All (Toàn bộ)** | **82.7%** | **78.5%** | **83.8%** | **57.6%** |
| cat (mèo) | 84.5% | 68.4% | 73.5% | 49.0% |
| chicken (gà) | 76.0% | 87.0% | 88.8% | 62.1% |
| cow (bò) | 80.7% | 73.0% | 76.3% | 52.7% |
| dog (chó) | 86.5% | 88.4% | 91.9% | 62.4% |
| horse (ngựa) | 82.9% | 87.5% | 93.2% | 66.0% |
| sheep (cừu) | 85.4% | 66.6% | 79.0% | 53.5% |

- **Preprocess Speed:** 1.2 ms / image
- **Inference Speed:** 73.4 ms / image (CPU)
- **Postprocess Speed:** 0.9 ms / image

*Results obtained by running `python val.py`*

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
├── dataset.py                  # Custom data augmentation script (NEW)
├── inference.py                # Inference with batch & visualization (UPDATED)
├── val.py                      # Validation script (NEW)
├── requirements.txt
├── yolov8n.pt                  # Pre-trained weights
├── checkpoints/
│   └── best.pt                 # Best performing model ⭐
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

