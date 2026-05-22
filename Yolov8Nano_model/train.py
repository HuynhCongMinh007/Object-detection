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
    print("[0/3] Kiểm tra tập dữ liệu huấn luyện...")
    train_images_dir = './data/train/images'
    if os.path.exists(train_images_dir):
        import glob
        img_files = [f for f in glob.glob(os.path.join(train_images_dir, '*')) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(img_files) < 4000:
            print(f"⚠️ Phát hiện số lượng ảnh train ({len(img_files)}) ít hơn 4000.")
            print("      Đang tiến hành tự động tăng cường dữ liệu...")
            try:
                from dataset import YOLOAugmenter
                augmenter = YOLOAugmenter(data_dir='./data')
                # Thực hiện tăng cường
                augmenter.generate_augmentations(target_count=4000)
                # Xuất ảnh trực quan hóa mẫu để kiểm định
                augmenter.save_visualizations(num_samples=10)
            except Exception as e:
                print(f"❌ Lỗi khi tự động tăng cường dữ liệu: {e}")
        else:
            print(f"✅ Tập dữ liệu train đã sẵn sàng với {len(img_files)} ảnh.")
    else:
        print(f"❌ Không tìm thấy thư mục ảnh train tại: {train_images_dir}")

    print(f"[1/3] Đang khởi tạo mô hình: {MODEL_TYPE}...")
    model = YOLO(MODEL_TYPE)


    import torch
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"[2/3] Bắt đầu huấn luyện với tập dữ liệu: {DATA_YAML}")
    print(f"      Thông số huấn luyện: Epochs = {opt.epochs} | Batch Size = {opt.batch_size}")
    print(f"      Thiết bị sử dụng: {device}")
    
    results = model.train(
        data=DATA_YAML,
        epochs=opt.epochs,
        batch=opt.batch_size,
        imgsz=IMG_SIZE,
        device=device, 
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