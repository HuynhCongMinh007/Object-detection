import os
import shutil
import argparse
from ultralytics import YOLO


DATA_YAML = './data/data.yaml'
MODEL_TYPE = 'yolov8n.pt'      
IMG_SIZE = 640
PROJECT_NAME = 'runs/train'
RUN_NAME = 'yolov8_animals'

def parse_opt():
    """Hàm phân tích các tham số từ dòng lệnh (CLI)"""
    parser = argparse.ArgumentParser(description="Huấn luyện YOLOv8 (Tinh gọn)")
    parser.add_argument('--epochs', type=int, default=50, help='Số vòng huấn luyện (mặc định: 50)')
    parser.add_argument('--batch-size', type=int, default=16, help='Kích thước batch (mặc định: 16)')
    return parser.parse_args()

def main(opt):
    print(f"[1/3] Đang khởi tạo mô hình: {MODEL_TYPE}...")
    model = YOLO(MODEL_TYPE)

    print(f"[2/3] Bắt đầu huấn luyện với tập dữ liệu: {DATA_YAML}")
    print(f"      Thông số huấn luyện: Epochs = {opt.epochs} | Batch Size = {opt.batch_size}")
    
    results = model.train(
        data=DATA_YAML,
        epochs=opt.epochs,
        batch=opt.batch_size,
        imgsz=IMG_SIZE,
        device='0', 
        project=PROJECT_NAME,
        name=RUN_NAME,
        exist_ok=True
    )
    
    print("[3/3] Huấn luyện hoàn tất!")
    
    run_dir = model.trainer.save_dir
    source_weight_path = os.path.join(run_dir, 'weights', 'best.pt')
    
    destination_dir = './checkpoints'
    destination_weight_path = os.path.join(destination_dir, 'best.pt')
    
    if os.path.exists(source_weight_path):
        os.makedirs(destination_dir, exist_ok=True)
        shutil.copy(source_weight_path, destination_weight_path)
        print(f"✅ Đã copy file trọng số tốt nhất vào: {destination_weight_path}")
    else:
        print(f"❌ Lỗi: Không tìm thấy file trọng số tại {source_weight_path}")

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)