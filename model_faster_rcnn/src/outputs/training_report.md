# BÁO CÁO HUẤN LUYỆN MÔ HÌNH FASTER R-CNN

## 1. Cấu hình Siêu tham số (Hyperparameters)
- **Backbone:** ResNet-50 FPN (Pretrained on MS COCO)
- **Số lượng lớp nhận diện:** 7 (6 động vật + 1 nền)
- **Thuật toán Tối ưu (Optimizer):** SGD (lr=0.005, momentum=0.9, weight_decay=0.0005)
- **Kích thước Batch (Batch Size):** 4
- **Cơ chế giảm LR:** Giảm khi Loss bão hòa (ReduceLROnPlateau, patience=3, factor=0.5)
- **Cơ chế dừng sớm (Early Stopping):** Tự động dừng sau 2 Epoch không cải thiện Loss

## 2. Kết quả Chung cuộc (Final Summary)
- **Trạng thái:** Bị dừng sớm bởi Early Stopping
- **Epoch đạt kết quả tốt nhất:** Epoch 9
- **Giá trị Validation Loss thấp nhất:** 0.1531
- **Trọng số lưu trữ tốt nhất tại:** `src/outputs/best_model.pth`

## 3. Nhật ký Huấn luyện chi tiết qua từng Epoch
| Epoch | Learning Rate | Train Loss | Valid Loss | RPN Cls Loss | RPN Box Loss | Head Cls Loss | Head Box Loss |
|---|---|---|---|---|---|---|---|
| 1 | 0.005000 | 0.3191 | 0.2188 | 0.0111 | 0.0178 | 0.1498 | 0.1404 |
| 2 | 0.005000 | 0.2009 | 0.1816 | 0.0054 | 0.0157 | 0.0859 | 0.0939 |
| 3 | 0.005000 | 0.1655 | 0.1709 | 0.0030 | 0.0148 | 0.0648 | 0.0828 |
| 4 | 0.005000 | 0.1528 | 0.1724 | 0.0028 | 0.0139 | 0.0598 | 0.0764 |
| 5 | 0.005000 | 0.1450 | 0.1638 | 0.0028 | 0.0134 | 0.0551 | 0.0738 |
| 6 | 0.005000 | 0.1416 | 0.1662 | 0.0028 | 0.0137 | 0.0547 | 0.0704 |
| 7 | 0.005000 | 0.1326 | 0.1576 | 0.0022 | 0.0134 | 0.0495 | 0.0675 |
| 8 | 0.005000 | 0.1252 | 0.1667 | 0.0023 | 0.0127 | 0.0449 | 0.0652 |
| 9 | 0.005000 | 0.1136 | 0.1531 | 0.0019 | 0.0125 | 0.0385 | 0.0606 |
| 10 | 0.005000 | 0.1075 | 0.1595 | 0.0020 | 0.0121 | 0.0345 | 0.0589 |
| 11 | 0.005000 | 0.1026 | 0.1597 | 0.0019 | 0.0118 | 0.0323 | 0.0566 |

## 4. Biểu đồ đường cong Loss (Loss Curve)
![Loss Curve](loss_curve.png)
