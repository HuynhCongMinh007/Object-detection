from ultralytics import YOLO
from pathlib import Path

# ==========================================
# FILE ĐÁNH GIÁ MÔ HÌNH (EVALUATION)
# Dùng để xuất Bảng mAP, Precision, Recall
# ==========================================

# 1. Đường dẫn tới file trọng số tốt nhất (nhớ dùng bản 24MB xịn nhé)
MODEL_PATH = './checkpoints/best.pt'
DATA_YAML = './data/data.yaml'

def main():
    # Kiểm tra checkpoint: nếu file mặc định không tồn tại, chọn file .pt gần nhất trong thư mục
    print(f">>> Kiểm tra checkpoint: {MODEL_PATH}")
    selected = Path(MODEL_PATH)
    if not selected.exists():
        ckpt_dir = selected.parent
        pts = sorted(list(ckpt_dir.glob('*.pt')), key=lambda p: p.stat().st_mtime, reverse=True)
        if pts:
            print(f">>> Không tìm thấy {selected.name}. Sử dụng checkpoint thay thế: {pts[0].name}")
            selected = pts[0]
        else:
            print(f"❌ Không tìm thấy file .pt trong {ckpt_dir}. Dừng chương trình.")
            return
    print(f">>> Đang tải mô hình từ: {selected}")
    model = YOLO(str(selected))

    print(f">>> Bắt đầu đánh giá mô hình trên tập TEST...")
    print(f">>> (Quá trình này sẽ đọc cả ảnh và file nhãn .txt để đối chiếu)\n")
    
    # Chạy hàm val() thay vì predict()
    metrics = model.val(
        data=DATA_YAML,
        split='test',     # Chỉ định chạy trên tập test trong data.yaml
        device='cpu',     # Đổi thành 0 nếu bạn chạy trên máy có GPU
        plots=True        # Cho phép YOLO tự động vẽ luôn các biểu đồ PR_curve, Confusion Matrix
    )
    
    print("\n>>> Đánh giá hoàn tất!")
    print(f">>> Bảng kết quả mAP và các biểu đồ chi tiết đã được lưu tại: {metrics.save_dir}")

if __name__ == '__main__':
    main()