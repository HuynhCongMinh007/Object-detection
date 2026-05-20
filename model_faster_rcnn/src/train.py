"""
train.py - Vòng lặp huấn luyện nâng cao cho Faster R-CNN (Task 4).
Hỗ trợ:
- Tự động giảm LR khi Loss bão hòa (ReduceLROnPlateau).
- Tự động dừng sớm khi quá khớp (Early Stopping) và hồi phục trọng số tốt nhất.
- Ghi nhật ký huấn luyện định dạng JSON chuẩn học sâu.
- Tự động tạo Báo cáo kết quả huấn luyện định dạng Markdown chuyên nghiệp.
"""
import os
import json
import time
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from config import *
from dataset import AnimalDataset, get_train_transforms, get_valid_transforms, collate_fn
from model import get_model


# ============================================================
# HÀM HUẤN LUYỆN 1 EPOCH
# ============================================================

def train_one_epoch(model, data_loader, optimizer, device, epoch):
    """
    Huấn luyện mô hình qua 1 epoch.
    Faster R-CNN ở chế độ train() tự động tính loss khi nhận cả images và targets.
    [Tham chiếu]: Xem chi tiết luồng huấn luyện tại 'training_faster_rcnn.md'.

    Returns:
        avg_loss (float): Loss trung bình của epoch.
        avg_components (dict): 4 thành phần loss trung bình.
    """
    # model.train() kích hoạt chế độ Huấn luyện (Mục 2 & 4 trong training_faster_rcnn.md).
    model.train()
    total_loss = 0
    
    loss_components = {
        'loss_classifier': 0,    # L_cls của Fast R-CNN Head (Multi-class Cross-Entropy)
        'loss_box_reg': 0,       # L_reg của Fast R-CNN Head (Class-specific Smooth L1)
        'loss_objectness': 0,    # L_cls nhị phân của RPN (Binary Cross-Entropy)
        'loss_rpn_box_reg': 0,   # L_reg của RPN (Smooth L1)
    }
    num_batches = len(data_loader)

    for batch_idx, (images, targets) in enumerate(data_loader):
        # Chuyển dữ liệu lên GPU/CPU
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass: model trả về dict chứa 4 thành phần loss
        loss_dict = model(images, targets)

        # Tổng loss = tổng 4 thành phần
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass & cập nhật trọng số
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Ghi nhận loss
        total_loss += losses.item()
        for key in loss_components:
            if key in loss_dict:
                loss_components[key] += loss_dict[key].item()

        # In tiến trình mỗi 50 batch
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == num_batches:
            print(f"  Epoch [{epoch+1}] Batch [{batch_idx+1}/{num_batches}] "
                  f"Loss: {losses.item():.4f}")

    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in loss_components.items()}
    return avg_loss, avg_components


# ============================================================
# HÀM ĐÁNH GIÁ (VALIDATION)
# ============================================================

@torch.no_grad()
def validate(model, data_loader, device):
    """
    Đánh giá mô hình trên tập validation.
    Giữ model.train() để Faster R-CNN trả về loss (thay vì predictions).
    Dùng torch.no_grad() để không tính gradient → tiết kiệm bộ nhớ.
    """
    model.train()
    total_loss = 0
    loss_components = {
        'loss_classifier': 0,
        'loss_box_reg': 0,
        'loss_objectness': 0,
        'loss_rpn_box_reg': 0,
    }
    num_batches = len(data_loader)

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        total_loss += losses.item()
        for key in loss_components:
            if key in loss_dict:
                loss_components[key] += loss_dict[key].item()

    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in loss_components.items()}
    return avg_loss, avg_components


# ============================================================
# TẠO BÁO CÁO KẾT QUẢ HUẤN LUYỆN ĐỊNH DẠNG MARKDOWN
# ============================================================

def generate_markdown_report(history, best_epoch, best_val_loss, early_stopped=False):
    """Tự động ghi ra báo cáo training_report.md chất lượng cao."""
    report_path = os.path.join(OUTPUT_DIR, 'training_report.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# BÁO CÁO HUẤN LUYỆN MÔ HÌNH FASTER R-CNN\n\n")
        f.write("## 1. Cấu hình Siêu tham số (Hyperparameters)\n")
        f.write(f"- **Backbone:** ResNet-50 FPN (Pretrained on MS COCO)\n")
        f.write(f"- **Số lượng lớp nhận diện:** {NUM_CLASSES} (6 động vật + 1 nền)\n")
        f.write(f"- **Thuật toán Tối ưu (Optimizer):** SGD (lr={LEARNING_RATE}, momentum={MOMENTUM}, weight_decay={WEIGHT_DECAY})\n")
        f.write(f"- **Kích thước Batch (Batch Size):** {BATCH_SIZE}\n")
        f.write(f"- **Cơ chế giảm LR:** Giảm khi Loss bão hòa (ReduceLROnPlateau, patience=3, factor=0.5)\n")
        f.write(f"- **Cơ chế dừng sớm (Early Stopping):** Tự động dừng sau 2 Epoch không cải thiện Loss\n\n")
        
        f.write("## 2. Kết quả Chung cuộc (Final Summary)\n")
        status = "Bị dừng sớm bởi Early Stopping" if early_stopped else "Huấn luyện đầy đủ"
        f.write(f"- **Trạng thái:** {status}\n")
        f.write(f"- **Epoch đạt kết quả tốt nhất:** Epoch {best_epoch}\n")
        f.write(f"- **Giá trị Validation Loss thấp nhất:** {best_val_loss:.4f}\n")
        f.write(f"- **Trọng số lưu trữ tốt nhất tại:** `src/outputs/best_model.pth`\n\n")
        
        f.write("## 3. Nhật ký Huấn luyện chi tiết qua từng Epoch\n")
        f.write("| Epoch | Learning Rate | Train Loss | Valid Loss | RPN Cls Loss | RPN Box Loss | Head Cls Loss | Head Box Loss |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        
        for i in range(len(history['epochs'])):
            ep = history['epochs'][i]
            lr = history['lr'][i]
            tl = history['train_loss'][i]
            vl = history['val_loss'][i]
            rpn_cls = history['train_rpn_cls_loss'][i]
            rpn_box = history['train_rpn_box_loss'][i]
            head_cls = history['train_head_cls_loss'][i]
            head_box = history['train_head_box_loss'][i]
            
            f.write(f"| {ep} | {lr:.6f} | {tl:.4f} | {vl:.4f} | {rpn_cls:.4f} | {rpn_box:.4f} | {head_cls:.4f} | {head_box:.4f} |\n")
            
        f.write("\n## 4. Biểu đồ đường cong Loss (Loss Curve)\n")
        f.write("![Loss Curve](loss_curve.png)\n")
        
    print(f"[Báo cáo]: Đã sinh báo cáo huấn luyện Markdown tại '{report_path}'")


# ============================================================
# SCRIPT CHÍNH
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Classes: {CLASS_NAMES}")
    print()

    # --- 1. Tải dữ liệu ---
    print("=" * 60)
    print("BƯỚC 1: TẢI DỮ LIỆU")
    print("=" * 60)

    train_dataset = AnimalDataset(
        os.path.join(DATASET_DIR, 'train'), transforms=get_train_transforms()
    )
    valid_dataset = AnimalDataset(
        os.path.join(DATASET_DIR, 'valid'), transforms=get_valid_transforms()
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True,
    )

    print(f"Train: {len(train_dataset)} images ({len(train_loader)} batches)")
    print(f"Valid: {len(valid_dataset)} images ({len(valid_loader)} batches)")
    print()

    # --- 2. Khởi tạo Model ---
    print("=" * 60)
    print("BƯỚC 2: KHỞI TẠO MÔ HÌNH")
    print("=" * 60)

    model = get_model(NUM_CLASSES)
    model.to(DEVICE)
    print()

    # --- 3. Lưu cấu hình Siêu tham số dự án (config.json) ---
    config_dict = {
        "model_architecture": "Faster R-CNN ResNet50 FPN",
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "early_stopping_patience": 2,
        "lr_patience": 3,
        "lr_factor": 0.5,
        "device": str(DEVICE)
    }
    config_path = os.path.join(OUTPUT_DIR, 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)
    print(f"[Cấu hình]: Đã lưu thông số thiết lập tại '{config_path}'")

    # --- 4. Cấu hình Optimizer & Callback Scheduler ---
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )
    
    # Sử dụng ReduceLROnPlateau (Giảm LR khi validation loss bão hòa) giống bài Lab mẫu của bạn
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    print(f"Optimizer: SGD (lr={LEARNING_RATE}, momentum={MOMENTUM})")
    print(f"Scheduler: ReduceLROnPlateau (patience=3, factor=0.5)")
    print()

    # --- 5. Vòng lặp huấn luyện nâng cao ---
    print("=" * 60)
    print("BƯỚC 3: BẮT ĐẦU HUẤN LUYỆN (TÍCH HỢP CALLBACKS)")
    print("=" * 60)

    best_val_loss = float('inf')
    best_epoch = 1
    patience_counter = 0  # Đếm phục vụ Early Stopping
    early_stopped = False
    
    # Khởi tạo đối tượng lịch sử để lưu nhật ký JSON giống bài Lab mẫu của bạn
    history = {
        'epochs': [],
        'lr': [],
        'train_loss': [],
        'val_loss': [],
        'train_rpn_cls_loss': [],
        'train_rpn_box_loss': [],
        'train_head_cls_loss': [],
        'train_head_box_loss': [],
        'val_rpn_cls_loss': [],
        'val_rpn_box_loss': [],
        'val_head_cls_loss': [],
        'val_head_box_loss': []
    }

    start_training_time = time.time()

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        # A. Huấn luyện
        train_loss, train_comp = train_one_epoch(
            model, train_loader, optimizer, DEVICE, epoch
        )

        # B. Đánh giá trên tập Valid
        val_loss, val_comp = validate(model, valid_loader, DEVICE)

        # C. Cập nhật Learning Rate dựa trên Validation Loss
        lr_scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - start_time

        # D. Điền dữ liệu vào History dict
        history['epochs'].append(epoch + 1)
        history['lr'].append(current_lr)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_rpn_cls_loss'].append(train_comp['loss_objectness'])
        history['train_rpn_box_loss'].append(train_comp['loss_rpn_box_reg'])
        history['train_head_cls_loss'].append(train_comp['loss_classifier'])
        history['train_head_box_loss'].append(train_comp['loss_box_reg'])
        history['val_rpn_cls_loss'].append(val_comp['loss_objectness'])
        history['val_rpn_box_loss'].append(val_comp['loss_rpn_box_reg'])
        history['val_head_cls_loss'].append(val_comp['loss_classifier'])
        history['val_head_box_loss'].append(val_comp['loss_box_reg'])

        # E. In kết quả trực quan
        print(f"\n{'='*60}")
        print(f"EPOCH [{epoch+1}/{NUM_EPOCHS}] | Thời gian: {elapsed:.1f}s | LR hiện tại: {current_lr:.6f}")
        print(f"  Train Loss: {train_loss:.4f} | Valid Loss: {val_loss:.4f}")
        print(f"  [RPN]   loss_cls: {train_comp['loss_objectness']:.4f} | loss_box: {train_comp['loss_rpn_box_reg']:.4f}")
        print(f"  [Head]  loss_cls: {train_comp['loss_classifier']:.4f} | loss_box: {train_comp['loss_box_reg']:.4f}")
        print(f"{'='*60}")

        # F. Lưu mô hình tốt nhất (Best Model Checkpoint) và cơ chế Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0  # Reset bộ đếm kiên nhẫn khi có cải thiện
            
            save_path = os.path.join(OUTPUT_DIR, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  >>> [CHECKPOINT]: Đã lưu mô hình tốt nhất mới (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  >>> [Kiên nhẫn]: Validation Loss không giảm trong {patience_counter}/2 epoch liên tục.")

        # Kích hoạt Dừng sớm nếu không có sự cải thiện sau 2 Epoch
        if patience_counter >= 2:
            print("\n" + "#" * 60)
            print(f"!!! [DỪNG SỚM - EARLY STOPPING KÍCH HOẠT TẠI EPOCH {epoch+1}] !!!")
            print("Mô hình không cải thiện được Loss trên tập Valid. Đang dừng để tránh quá khớp (Overfitting).")
            print("#" * 60)
            early_stopped = True
            break

    # G. Khôi phục trọng số tốt nhất trước khi kết thúc (giống restore_best_weights=True của Keras)
    best_weights_path = os.path.join(OUTPUT_DIR, 'best_model.pth')
    if os.path.exists(best_weights_path):
        model.load_state_dict(torch.load(best_weights_path, map_location=DEVICE))
        print(f"\n[Hồi phục]: Đã khôi phục trọng số tối ưu nhất ở Epoch {best_epoch} vào mô hình.")

    # H. Lưu trọng số phiên bản cuối cùng
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'last_model.pth'))
    
    total_training_time = time.time() - start_training_time
    print(f"\nHuấn luyện hoàn tất trong {total_training_time/60:.1f} phút!")

    # --- 6. Ghi lịch sử huấn luyện lịch sử ra file JSON (training_history.json) ---
    history_path = os.path.join(OUTPUT_DIR, 'training_history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4, ensure_ascii=False)
    print(f"[Nhật ký]: Đã lưu chi tiết quá trình huấn luyện tại '{history_path}'")

    # --- 7. Vẽ và lưu đồ thị đường cong Loss ---
    plt.figure(figsize=(10, 5))
    plt.plot(history['epochs'], history['train_loss'], 'b-o', linewidth=2, label='Train Loss')
    plt.plot(history['epochs'], history['val_loss'], 'r-o', linewidth=2, label='Valid Loss')
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Loss', fontsize=11)
    plt.title('Training & Validation Loss Curve', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    curve_path = os.path.join(OUTPUT_DIR, 'loss_curve.png')
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"[Đồ thị]: Đã xuất biểu đồ Loss tại '{curve_path}'")

    # --- 8. Tự động xuất Báo cáo Markdown ---
    generate_markdown_report(history, best_epoch, best_val_loss, early_stopped)


if __name__ == '__main__':
    main()
