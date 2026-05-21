"""
dataset.py - Custom Dataset class với Data Augmentation cho bộ dữ liệu Animals.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import CLASS_TO_IDX


# ============================================================
# DATA AUGMENTATION PIPELINES
# ============================================================

def get_train_transforms():
    """
    Pipeline augmentation cho tập train sử dụng thư viện albumentations.

    Các phép biến đổi:
    - HorizontalFlip: Lật ngang ảnh (mô phỏng góc nhìn khác).
    - ColorJitter: Thay đổi ngẫu nhiên độ sáng, tương phản, bão hòa.
    - GaussianBlur: Làm mờ nhẹ (tăng khả năng tổng quát hóa).

    Lưu ý: albumentations tự động biến đổi tọa độ bounding boxes
    tương ứng khi ảnh bị lật/crop nhờ tham số bbox_params.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(
            brightness=0.3, contrast=0.3,
            saturation=0.3, hue=0.1, p=0.5
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        # Chuyển numpy HWC → tensor CHW
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',         # Format tọa độ: (xmin, ymin, xmax, ymax)
        label_fields=['labels'],     # Tên trường chứa nhãn tương ứng
        min_visibility=0.3,          # Loại box nếu diện tích hiển thị < 30%
        clip=True,                   # Tự động clip tọa độ ngoài biên để tránh lỗi làm tròn
    ))


def get_valid_transforms():
    """Pipeline cho tập validation/test: chỉ chuyển sang tensor, không augment."""
    return A.Compose([
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.3,
        clip=True,                   # Tự động clip tọa độ ngoài biên để tránh lỗi làm tròn
    ))


# ============================================================
# CUSTOM DATASET CLASS
# ============================================================

class AnimalDataset(Dataset):
    """
    Custom Dataset cho bộ dữ liệu Animals Object Detection.

    Quy trình xử lý:
    1. Đọc file _annotations.csv chứa thông tin bounding boxes.
    2. Gom nhóm (group) các boxes có cùng filename thành 1 mẫu.
    3. Khi lấy mẫu (__getitem__), đọc ảnh + áp dụng augmentation.
    4. Trả về (image_tensor, target_dict) theo format của Faster R-CNN.
    """

    def __init__(self, root_dir, transforms=None):
        """
        Args:
            root_dir (str): Đường dẫn thư mục chứa ảnh và _annotations.csv.
            transforms: Pipeline augmentation (albumentations.Compose).
        """
        self.root_dir = root_dir
        self.transforms = transforms

        # Đọc file CSV
        csv_path = os.path.join(root_dir, '_annotations.csv')
        df = pd.read_csv(csv_path)
        df = df.dropna()  # Loại bỏ dòng trống (nếu có)

        # === Gom nhóm các bounding boxes có cùng filename ===
        # Mỗi ảnh có thể chứa nhiều đối tượng → nhiều dòng trong CSV
        self.image_ids = df['filename'].unique().tolist()
        self.annotations = {}

        for img_name in self.image_ids:
            img_df = df[df['filename'] == img_name]
            
            sanitized_boxes = []
            sanitized_labels = []
            
            for _, row in img_df.iterrows():
                w = float(row['width'])
                h = float(row['height'])
                xmin = float(row['xmin'])
                ymin = float(row['ymin'])
                xmax = float(row['xmax'])
                ymax = float(row['ymax'])
                cls = row['class']
                
                # 1. Clip tọa độ vào trong giới hạn ảnh để tránh lỗi Albumentations
                xmin = max(0.0, min(xmin, w - 1.0))
                ymin = max(0.0, min(ymin, h - 1.0))
                xmax = max(xmin + 1.0, min(xmax, w))
                ymax = max(ymin + 1.0, min(ymax, h))
                
                # 2. Chỉ giữ lại các hộp giới hạn hợp lệ có diện tích dương
                if xmin < xmax and ymin < ymax:
                    sanitized_boxes.append([xmin, ymin, xmax, ymax])
                    sanitized_labels.append(CLASS_TO_IDX[cls])
                    
            self.annotations[img_name] = {
                'boxes': sanitized_boxes,
                'labels': sanitized_labels,
            }

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # === 1. Đọc ảnh ===
        img_name = self.image_ids[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = np.array(Image.open(img_path).convert('RGB'))  # HWC, uint8

        # === 2. Lấy annotations (boxes + labels) ===
        ann = self.annotations[img_name]
        boxes = ann['boxes']
        labels = ann['labels']

        # === 3. Áp dụng Data Augmentation ===
        if self.transforms is not None:
            transformed = self.transforms(
                image=image,
                bboxes=boxes,
                labels=labels,
            )
            image = transformed['image']      # Tensor CHW, uint8
            boxes = transformed['bboxes']
            labels = transformed['labels']
        else:
            # Nếu không có transforms, chuyển thủ công sang tensor
            image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1)

        # === 4. Chuẩn hóa ảnh về [0, 1] ===
        # Faster R-CNN yêu cầu input là float32 trong khoảng [0, 1]
        # (Model tự xử lý normalize với ImageNet mean/std bên trong)
        image = image.float() / 255.0

        # === 5. Chuyển đổi annotations sang tensor ===
        if len(boxes) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)

        # Tính diện tích mỗi box (cần thiết cho Faster R-CNN)
        if boxes_tensor.numel() > 0:
            area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * \
                   (boxes_tensor[:, 3] - boxes_tensor[:, 1])
        else:
            area = torch.zeros((0,), dtype=torch.float32)

        # iscrowd = 0 cho tất cả (không có annotation dạng crowd)
        iscrowd = torch.zeros(len(labels_tensor), dtype=torch.int64)

        target = {
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'image_id': torch.tensor([idx]),
            'area': area,
            'iscrowd': iscrowd,
        }

        return image, target


def collate_fn(batch):
    """
    Custom collate function cho DataLoader.
    Faster R-CNN nhận list các ảnh (kích thước có thể khác nhau),
    không stack thành 1 batch tensor như classification.
    """
    return tuple(zip(*batch))
