"""
nms.py - Triển khai thủ công thuật toán Non-Maximum Suppression (NMS).
KHÔNG sử dụng torchvision.ops.nms.
"""
import torch


def compute_iou(box_a, boxes_b):
    """
    Tính Intersection over Union (IoU) giữa 1 box và N boxes khác.
    [Tham chiếu]: Đối chiếu với toán học IoU tại Mục 2.C trong 'training_faster_rcnn.md'.

    Công thức IoU:
        IoU = Diện tích giao nhau (Intersection) / Diện tích hợp (Union)
        Union = Area_A + Area_B - Intersection

    Args:
        box_a (Tensor): 1 bounding box, shape [4], format (xmin, ymin, xmax, ymax).
        boxes_b (Tensor): N bounding boxes, shape [N, 4].

    Returns:
        Tensor: IoU values, shape [N].
    """
    # Tọa độ của box_a
    a_x1, a_y1, a_x2, a_y2 = box_a[0], box_a[1], box_a[2], box_a[3]

    # Tọa độ của các boxes_b
    b_x1 = boxes_b[:, 0]
    b_y1 = boxes_b[:, 1]
    b_x2 = boxes_b[:, 2]
    b_y2 = boxes_b[:, 3]

    # --- Tính tọa độ vùng giao nhau (Intersection) ---
    # Góc trên-trái của vùng giao = max(góc trên-trái của A, góc trên-trái của B)
    inter_x1 = torch.max(a_x1, b_x1)
    inter_y1 = torch.max(a_y1, b_y1)
    # Góc dưới-phải của vùng giao = min(góc dưới-phải của A, góc dưới-phải của B)
    inter_x2 = torch.min(a_x2, b_x2)
    inter_y2 = torch.min(a_y2, b_y2)

    # Chiều rộng và chiều cao vùng giao (clamp >= 0 vì 2 box có thể không giao nhau)
    inter_width = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_height = torch.clamp(inter_y2 - inter_y1, min=0)

    # Diện tích vùng giao = chiều rộng × chiều cao
    intersection = inter_width * inter_height

    # --- Tính diện tích từng box ---
    area_a = (a_x2 - a_x1) * (a_y2 - a_y1)
    area_b = (b_x2 - b_x1) * (b_y2 - b_y1)

    # --- Tính diện tích hợp (Union) ---
    # Union = Area_A + Area_B - Intersection (trừ phần giao để không đếm 2 lần)
    union = area_a + area_b - intersection

    # --- Tính IoU, cộng epsilon nhỏ để tránh chia cho 0 ---
    iou = intersection / (union + 1e-6)

    return iou


def custom_nms(boxes, scores, iou_threshold):
    """
    Non-Maximum Suppression (NMS) triển khai thủ công từ đầu.
    [Tham chiếu]: Đối chiếu các bước NMS tại Mục 2.E (training_faster_rcnn.md) 
                 hoặc Mục 2.D & 4.C (inference_faster_rcnn.md).

    Mục đích: Khi mô hình dự đoán, nó thường tạo ra nhiều bounding boxes
    chồng chéo nhau cho cùng một đối tượng. NMS loại bỏ các box trùng lặp,
    chỉ giữ lại box có điểm tin cậy (confidence score) cao nhất.

    Thuật toán Greedy NMS:
        1. Sắp xếp tất cả box theo score giảm dần.
        2. Chọn box có score cao nhất → thêm vào danh sách "giữ lại" (keep).
        3. Tính IoU giữa box vừa chọn với tất cả box còn lại.
        4. Loại bỏ các box có IoU > ngưỡng (vì chúng phát hiện cùng 1 đối tượng).
        5. Lặp lại bước 2-4 cho đến khi không còn box nào.

    Args:
        boxes (Tensor): Tọa độ bounding boxes, shape [N, 4],
                        format (xmin, ymin, xmax, ymax).
        scores (Tensor): Điểm tin cậy của mỗi box, shape [N].
        iou_threshold (float): Ngưỡng IoU. Nếu IoU > ngưỡng → loại bỏ box.

    Returns:
        Tensor: Chỉ số (indices) của các box được giữ lại, shape [M] với M <= N.
    """
    # Trường hợp đặc biệt: không có box nào → trả về rỗng
    if boxes.numel() == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device)

    # === BƯỚC 1: Sắp xếp score giảm dần, lấy chỉ số ===
    # Ví dụ: scores = [0.8, 0.95, 0.6] → order = [1, 0, 2]
    _, order = scores.sort(descending=True)

    keep = []  # Danh sách chỉ số các box được giữ lại

    # === BƯỚC 2: Vòng lặp Greedy NMS ===
    while order.numel() > 0:
        # Lấy index của box có score cao nhất (đứng đầu danh sách)
        current_idx = order[0].item()
        keep.append(current_idx)

        # Nếu chỉ còn 1 box → dừng vòng lặp
        if order.numel() == 1:
            break

        # Lấy danh sách các box còn lại (bỏ box vừa chọn)
        remaining_indices = order[1:]

        # === BƯỚC 3: Tính IoU giữa box hiện tại và tất cả box còn lại ===
        current_box = boxes[current_idx]
        remaining_boxes = boxes[remaining_indices]
        iou = compute_iou(current_box, remaining_boxes)

        # === BƯỚC 4: Loại bỏ box có IoU > ngưỡng ===
        # Giữ lại các box có IoU <= ngưỡng (tức là KHÔNG trùng lặp với box hiện tại)
        # Ví dụ: iou = [0.8, 0.2, 0.9], threshold = 0.5
        #   → mask = [False, True, False] → chỉ giữ box có iou = 0.2
        mask = iou <= iou_threshold
        order = remaining_indices[mask]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


# ============================================================
# KIỂM TRA NHANH (Unit Test)
# ============================================================
if __name__ == '__main__':
    print("=== Kiểm tra hàm Custom NMS ===")

    # Tạo dữ liệu mẫu: 4 boxes, trong đó box 0-1 gần trùng, box 2-3 gần trùng
    test_boxes = torch.tensor([
        [100, 100, 210, 210],   # Box 0
        [105, 105, 215, 215],   # Box 1 (gần trùng box 0)
        [300, 300, 400, 400],   # Box 2
        [305, 305, 405, 405],   # Box 3 (gần trùng box 2)
    ], dtype=torch.float32)
    test_scores = torch.tensor([0.9, 0.75, 0.8, 0.6])

    kept = custom_nms(test_boxes, test_scores, iou_threshold=0.5)
    print(f"Input: {len(test_scores)} boxes")
    print(f"Output (kept indices): {kept.tolist()}")
    print(f"Kỳ vọng: [0, 2] (giữ box 0 và box 2, loại box 1 và box 3 vì trùng)")

    # Test IoU riêng
    iou_val = compute_iou(test_boxes[0], test_boxes[1:2])
    print(f"\nIoU giữa box 0 và box 1: {iou_val.item():.4f}")
