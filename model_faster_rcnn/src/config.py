"""
config.py - Cấu hình siêu tham số và hằng số cho dự án.
"""
import torch

# ============================================================
# ĐƯỜNG DẪN DỮ LIỆU
# ============================================================
# Đường dẫn tới thư mục gốc chứa 3 thư mục con: train, valid, test
DATASET_DIR = './dataset'

# Đường dẫn lưu model
OUTPUT_DIR = './outputs'

# ============================================================
# SIÊU THAM SỐ HUẤN LUYỆN
# ============================================================
NUM_CLASSES = 7       # 6 động vật + 1 background
BATCH_SIZE = 4
NUM_WORKERS = 2
NUM_EPOCHS = 15
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
LR_STEP_SIZE = 5      # Giảm LR sau mỗi N epoch
LR_GAMMA = 0.5        # Hệ số giảm LR

# ============================================================
# SIÊU THAM SỐ INFERENCE
# ============================================================
NMS_IOU_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.5

# ============================================================
# THIẾT BỊ TÍNH TOÁN
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# ÁNH XẠ LỚP (CLASS MAPPING)
# ============================================================
# Index 0 dành cho background (bắt buộc bởi Faster R-CNN)
CLASS_NAMES = ['__background__', 'cat', 'chicken', 'cow', 'dog', 'horse', 'sheep']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Bảng màu để vẽ bounding box cho từng lớp
COLORS = ['', '#FF6B6B', '#FECA57', '#48DBFB', '#FF9FF3', '#54A0FF', '#5F27CD']
