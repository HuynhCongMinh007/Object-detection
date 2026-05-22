from ultralytics import YOLO
from pathlib import Path

# ==========================================
# FILE ĐÁNH GIÁ MÔ HÌNH (EVALUATION)
# Dùng để xuất Bảng mAP, Precision, Recall
# ==========================================

MODEL_PATH = './checkpoints/best.pt'
DATA_YAML = './data/data.yaml'

def main():
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
    
    import torch
    device = '0' if torch.cuda.is_available() else 'cpu'
    metrics = model.val(
        data=DATA_YAML,
        split='test',     
        device=device,     
        plots=True       
    )
    print("\n>>> Đánh giá hoàn tất!")
    print(f">>> Bảng kết quả mAP và các biểu đồ chi tiết đã được lưu tại: {metrics.save_dir}")

if __name__ == '__main__':
    main()