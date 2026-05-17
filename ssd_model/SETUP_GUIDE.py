"""
GPU Setup & Training Guide for RTX 3050 (4GB VRAM)
NVIDIA GeForce RTX 3050 with CUDA 13.0 optimized configuration
"""

# ============================================================================
# COMPLETE SETUP & TRAINING GUIDE
# ============================================================================
#
# This guide walks through setting up and training the SSD300 model on
# RTX 3050 4GB GPU with your COCO-format animal dataset.
#
# ============================================================================
# PART 1: INSTALLATION (One-time setup)
# ============================================================================
#
# Step 1a: Open PowerShell and navigate to ssd_model folder
#
# cd "C:\Users\Admin\Desktop\HỌC THỐNG KÊ\Thực hành\Final\Object-detection\ssd_model"
#
#
# Step 1b: Create Virtual Environment
#
# python -m venv .venv
#
#
# Step 1c: Activate Virtual Environment
#
# .venv\Scripts\activate
#
# (You should see "(.venv)" prefix in your PowerShell prompt)
#
#
# Step 1d: Upgrade pip and setuptools
#
# python -m pip install --upgrade pip setuptools wheel
#
#
# Step 1e: Install PyTorch with CUDA 12.1 support
#        [RTX 3050 supports CUDA 12.x; check pytorch.org/get-started for latest]
#
# Option A - CUDA 12.1 (Recommended for CUDA 13.0):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#
# Option B - If above doesn't work, try CUDA 11.8:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#
# [If both fail, visit https://pytorch.org/get-started/locally/ and select:
#  OS=Windows, Package=pip, Language=Python, CUDA=<your-cuda-version>]
#
#
# Step 1f: Install project dependencies
#
# pip install -r requirements.txt
#
#
# Step 1g: Verify GPU is detected
#
# python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
#
# Expected output:
# CUDA available: True
# Device: NVIDIA GeForce RTX 3050
#
#
# ============================================================================
# PART 2: DATASET PREPARATION
# ============================================================================
#
# Your dataset structure should be:
#
# animal_dataset/
# └── train/
#     ├── images/
#     │   ├── image_001.jpg
#     │   ├── image_002.jpg
#     │   └── ... (all training images)
#     └── _annotations.coco.json
#
#
# The _annotations.coco.json file should contain:
# {
#   "images": [...],
#   "annotations": [...],
#   "categories": [
#     {"id": 1, "name": "animal_1"},
#     {"id": 2, "name": "animal_2"},
#     ... (up to 6 animal classes)
#   ]
# }
#
# IMPORTANT NOTES:
# - Class indices in COCO JSON start at 1 (not 0)
# - Index 0 is reserved for background class (use by model automatically)
# - Total classes = 6 animals + 1 background = 7 classes (NUM_CLASSES=7 in train.py)
#
#
# ============================================================================
# PART 3: VERIFY DATASET BEFORE TRAINING
# ============================================================================
#
# Step 3a: Run the dataset test
#
# python test_dataset.py
#
# This will:
# ✓ Load your COCO JSON and verify the format
# ✓ Check image paths and annotations
# ✓ Print class names and mappings
# ✓ Load one sample and verify data types
# ✓ Check GPU availability
#
# Expected output (partial):
#
# ================================================================================
# COCO Dataset Loading Test
# ================================================================================
#
# [1/3] Loading dataset...
# [AnimalDataset] Loaded COCO JSON from: ./animal_dataset/train/_annotations.coco.json
# [AnimalDataset] Found 6 animal classes:
#   1: cat
#   2: dog
#   3: bird
#   4: fox
#   5: rabbit
#   6: ...
#
# [2/3] Dataset properties:
#   Total samples: 500
#   Total classes (including background): 7
#
# [3/3] Loading sample...
# ✓ Sample loaded successfully!
#   Image shape: torch.Size([3, 300, 300])
#   Number of objects: 3
#
# If you see any ERRORS here, your dataset has issues. Debug before training.
#
#
# ============================================================================
# PART 4: DOWNLOAD PRETRAINED WEIGHTS (Optional but Recommended)
# ============================================================================
#
# This step downloads the SSD300-VGG16 COCO pretrained weights (~100MB)
# and saves them locally to avoid repeated downloads from the internet.
#
# Step 4a: Run download script
#
# python download_weights.py
#
# Expected output:
# Loading SSD300-VGG16 pretrained weights from torchvision...
# Saving pretrained state dict to: ./checkpoints/ssd300_vgg16_coco_pretrained.pth
# Done.
#
# This creates: ./checkpoints/ssd300_vgg16_coco_pretrained.pth (~100MB)
#
# Note: If you skip this step, train.py will automatically download weights
#       on first run (requires internet connection).
#
#
# ============================================================================
# PART 5: TRAINING
# ============================================================================
#
# Step 5a: Start training with GPU
#
# python train.py --epochs 100 --batch-size 4 --lr 0.001
#
#
# WHAT THE COMMAND MEANS:
# - --epochs 100 : Train for 100 epochs (iterations over full dataset)
# - --batch-size 4 : Process 4 images at a time (optimized for RTX 3050 4GB)
# - --lr 0.001 : Learning rate 0.001 (lower = slower but more stable learning)
#
#
# EXPECTED CONSOLE OUTPUT:
#
# Device: cuda:0
# Num Classes (with background): 7
# Batch Size: 4
# Learning Rate: 0.001
#
# [1/4] Loading datasets...
# [AnimalDataset] Loaded COCO JSON from: ./animal_dataset/train/_annotations.coco.json
# [AnimalDataset] Found 6 animal classes:
# ...
# Training samples: 500
#
# [2/4] Creating data loaders...
# [3/4] Loading SSD300-VGG16 model (pretrained COCO weights)...
# Loaded SSD300-VGG16 with torchvision DEFAULT (COCO) weights
# Modifying classification head for 7 classes...
# Model ready for training
#
# [4/4] Setting up optimizer and scheduler...
#
# ================================================================================
# Starting training...
# ================================================================================
#
# Epoch [1/100]
#   Epoch [1] Batch [50/125] Loss: 2.1234 (Avg: 2.4567)
#   Epoch [1] Batch [100/125] Loss: 1.8912 (Avg: 2.2341)
#   Train Loss: 2.2156
#   Val Loss: 1.9876
#   ✓ Best model saved to ./checkpoints/best_ssd.pth
#
# Epoch [2/100]
#   Epoch [2] Batch [50/125] Loss: 1.7654 (Avg: 1.9023)
#   ...
#
#
# TRAINING SPEED (RTX 3050 with batch_size=4):
# - Approximately 30-60 seconds per epoch (depends on dataset size)
# - For 500 training samples: ~4-8 minutes per epoch
# - Full training (100 epochs): ~7-13 hours
#
#
# OUTPUT FILES:
# - ./checkpoints/best_ssd.pth : Best model (lowest validation loss)
# - Shows loss metrics during training
#
#
# ============================================================================
# PART 6: TROUBLESHOOTING
# ============================================================================
#
# PROBLEM: "CUDA out of memory"
# SOLUTION: Reduce batch size in train.py
#   - Try: BATCH_SIZE = 2 (instead of 4)
#   - Or close other GPU-intensive apps (VS Code, browsers, etc.)
#
#
# PROBLEM: "ModuleNotFoundError: No module named 'torch'"
# SOLUTION: Activate virtual environment and reinstall PyTorch
#   .venv\Scripts\activate
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#
#
# PROBLEM: "CUDA is not available"
# SOLUTION: Check NVIDIA driver and CUDA toolkit
#   nvidia-smi          # Check driver version
#   Check pytorch.org for compatible CUDA version
#
#
# PROBLEM: "FileNotFoundError: Annotation file not found"
# SOLUTION: Update paths in train.py line 18-19
#   TRAIN_IMG_DIR = './animal_dataset/train'
#   TRAIN_ANN_FILE = './animal_dataset/train/_annotations.coco.json'
#
#
# PROBLEM: "Empty dataset"
# SOLUTION: Check COCO JSON format and class indices
#   Run: python test_dataset.py  # See detailed error messages
#
#
# ============================================================================
# PART 7: AFTER TRAINING
# ============================================================================
#
# Once training completes, you'll have:
# - ./checkpoints/best_ssd.pth : Your trained model
#
# To use the model for inference:
#
# python -c "
# from inference import load_model, run_inference
# import torch
#
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = load_model('./checkpoints/best_ssd.pth', num_classes=7, device=device)
#
# result = run_inference(
#     model,
#     'path/to/test/image.jpg',
#     idx_to_class={1: 'cat', 2: 'dog', 3: 'bird', 4: 'fox', 5: 'rabbit', 6: 'class_6'},
#     device=device
# )
#
# print(f'Found {result[\"num_detections\"]} objects')
# for det in result['detections']:
#     print(f'  {det[\"label\"]}: {det[\"confidence\"]:.2f}')
# "
#
#
# ============================================================================
# PART 8: CONFIGURATION SUMMARY FOR RTX 3050
# ============================================================================
#
# ✓ Device: CUDA (RTX 3050)
# ✓ GPU Memory: 4GB
# ✓ Default batch_size: 4 (avoid OOM)
# ✓ Num classes: 7 (6 animals + background)
# ✓ Model: SSD300-VGG16 (pre-trained on COCO)
# ✓ Transfer learning: Yes (faster convergence)
# ✓ Optimizer: SGD (momentum=0.9, weight_decay=5e-4)
# ✓ LR scheduler: Step decay (every 30 epochs, gamma=0.1)
#
#
# ============================================================================
# QUICK START (Summary)
# ============================================================================
#
# 1. cd ssd_model
# 2. python -m venv .venv
# 3. .venv\Scripts\activate
# 4. pip install -r requirements.txt
# 5. pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# 6. python test_dataset.py                  # Verify dataset
# 7. python download_weights.py              # Download pretrained weights
# 8. python train.py --epochs 100            # Start training
#
#
# Total time to setup: ~10 minutes (including package downloads)
# Total time to train (100 epochs on 500 images): ~7-13 hours
#
# ============================================================================

print(__doc__)
