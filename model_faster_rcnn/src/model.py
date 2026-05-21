from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model(num_classes):
    """
    Khởi tạo mô hình Faster R-CNN với backbone ResNet50-FPN pretrained trên COCO.
    Thay đổi lớp box_predictor ở phần ROI Head cho phù hợp số classes dự án.

    Kiến trúc Faster R-CNN gồm 3 phần chính:
    ┌─────────────────────────────────────────────────┐
    │  1. Backbone (ResNet50 + FPN)                   │
    │     → Trích xuất feature maps đa tỷ lệ từ ảnh  │
    ├─────────────────────────────────────────────────┤
    │  2. Region Proposal Network (RPN)               │
    │     → Đề xuất ~2000 vùng có thể chứa đối tượng │
    ├─────────────────────────────────────────────────┤
    │  3. ROI Head (Box Predictor)                    │
    │     → Phân loại + hồi quy bounding box chính xác│
    └─────────────────────────────────────────────────┘

    Args:
        num_classes (int): Tổng số lớp (bao gồm background).
                          Dự án: 7 = 6 động vật + 1 background.

    Returns:
        model: Mô hình Faster R-CNN đã được chỉnh sửa head.
    """
    # Tải Faster R-CNN pretrained trên COCO (91 classes)
    # [Tham chiếu]: Mục 1 (Backbone) trong training_faster_rcnn.md & inference_faster_rcnn.md
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    # Lấy số features đầu vào của lớp box_predictor hiện tại (D = 1024 cho ResNet50-FPN)
    # [Tham chiếu]: Mục 4.A (Trích xuất đặc trưng f_m) trong 2 file mô tả
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Thay thế box_predictor bằng lớp mới với đúng số classes (Học chuyển vị - Transfer Learning)
    # FastRCNNPredictor gồm 2 nhánh song song (Mục 4.B & 4.C trong cả 2 file):
    #   - cls_score: Lớp Tuyến tính phân loại (Linear) đầu ra K+1 classes (Mục 4.B - Nhánh Phân Loại Lớp)
    #   - bbox_pred: Lớp Tuyến tính hồi quy (Linear) đầu ra (K+1)*4 tọa độ (Mục 4.B - Nhánh Tinh Chỉnh Cấp Lớp)
    #                Đầu ra là (K+1)*4 để thực hiện Class-specific Regression (nắn hộp riêng biệt cho từng lớp).
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    print(f"[Model] Faster R-CNN initialized")
    print(f"  Backbone: ResNet50-FPN (pretrained on COCO)")
    print(f"  Box predictor: {in_features} features → {num_classes} classes")

    return model
