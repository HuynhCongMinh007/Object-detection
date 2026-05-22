import cv2
import json
import argparse
from pathlib import Path
from ultralytics import YOLO

# Cấu hình
MODEL_PATH = './checkpoints/best.pt'
CONFIDENCE_THRESHOLD = 0.5

CLASS_NAMES = ['cat', 'chicken', 'cow', 'dog', 'horse', 'sheep']

def load_model(model_path=MODEL_PATH):
    """Tải mô hình YOLOv8"""
    print(f">>> Đang tải mô hình từ: {model_path}")
    return YOLO(model_path)

def run_inference(model, image_path, conf_threshold=CONFIDENCE_THRESHOLD):
    """
    Chạy dự đoán và trả về format dictionary y hệt mô hình SSD của nhóm.
    """
    print(f">>> Đang xử lý ảnh: {image_path}")
    
    results = model.predict(source=image_path, conf=conf_threshold, save=False)
    result = results[0] 

    output = {
        'detections': [],
        'image_size': [result.orig_shape[0], result.orig_shape[1]], # [height, width]
        'num_detections': 0
    }
    
    # Trích xuất dữ liệu từ kết quả YOLOv8
    for box in result.boxes:
        # Tọa độ [x1, y1, x2, y2]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = CLASS_NAMES[cls_id]
        
        output['detections'].append({
            'bbox': [int(x1), int(y1), int(x2), int(y2)],
            'label': label,
            'confidence': round(conf, 4)
        })
        
    output['num_detections'] = len(output['detections'])
    
    # Lưu annotated image (với bounding box)
    output['annotated_image_path'] = None
    if result.plot() is not None:
        output['annotated_image_path'] = str(image_path)  
    
    return output

def draw_and_save_predictions(image_path, detections, output_dir, image_size):
    """
    Vẽ bounding box lên ảnh và lưu vào thư mục predictions
    """
    # Đọc ảnh gốc
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"⚠ Không thể đọc ảnh: {image_path}")
        return None
    
    # Màu sắc cho từng class
    colors = {
        'cat': (0, 255, 0),        # Green
        'chicken': (255, 0, 0),    # Blue
        'cow': (0, 0, 255),        # Red
        'dog': (255, 255, 0),      # Cyan
        'horse': (255, 0, 255),    # Magenta
        'sheep': (0, 165, 255)     # Orange
    }
    
    # Vẽ từng detection
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        label = det['label']
        conf = det['confidence']
        
        color = colors.get(label, (255, 255, 255))
        
        # Vẽ bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Vẽ label
        text = f"{label} ({conf:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # Background cho text
        cv2.rectangle(img, (x1, y1 - text_size[1] - 6), 
                     (x1 + text_size[0] + 6, y1), color, -1)
        cv2.putText(img, text, (x1 + 3, y1 - 3), font, 
                   font_scale, (0, 0, 0), thickness)
    
    # Tạo tên file output
    img_stem = Path(image_path).stem
    output_path = Path(output_dir) / 'predictions' / f"{img_stem}_pred.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Lưu ảnh
    cv2.imwrite(str(output_path), img)
    return str(output_path)

def print_summary(all_results, conf_threshold):
    """
    In ra tóm tắt thống kê kết quả inference
    """
    print("\n" + "="*70)
    print("📊 TÓM TẮT KẾT QUẢ INFERENCE")
    print("="*70)
    
    # Thống kê chung
    total_images = len(all_results)
    images_with_detections = sum(1 for r in all_results.values() if r['num_detections'] > 0)
    images_without_detections = total_images - images_with_detections
    total_detections = sum(r['num_detections'] for r in all_results.values())
    
    print(f"\n📷 TỔNG QUAN:")
    print(f"   • Tổng số ảnh: {total_images}")
    print(f"   • Ảnh có phát hiện: {images_with_detections} ({images_with_detections/total_images*100:.1f}%)")
    print(f"   • Ảnh không có phát hiện: {images_without_detections}")
    print(f"   • Tổng số detections: {total_detections}")
    print(f"   • Confidence threshold: {conf_threshold}")
    
    # Thống kê per class
    class_stats = {}
    for class_name in CLASS_NAMES:
        class_stats[class_name] = {
            'count': 0,
            'confidences': []
        }
    
    for result in all_results.values():
        for det in result['detections']:
            label = det['label']
            conf = det['confidence']
            class_stats[label]['count'] += 1
            class_stats[label]['confidences'].append(conf)
    
    print(f"\n🐾 THỐNG KÊ THEO CLASS:")
    print(f"{'Class':<15} {'Count':<10} {'Avg Conf':<12} {'Min':<10} {'Max':<10}")
    print("-" * 70)
    
    for class_name in CLASS_NAMES:
        stats = class_stats[class_name]
        if stats['count'] > 0:
            avg_conf = sum(stats['confidences']) / len(stats['confidences'])
            min_conf = min(stats['confidences'])
            max_conf = max(stats['confidences'])
            print(f"{class_name:<15} {stats['count']:<10} {avg_conf:<12.4f} {min_conf:<10.4f} {max_conf:<10.4f}")
        else:
            print(f"{class_name:<15} {0:<10} {'N/A':<12} {'N/A':<10} {'N/A':<10}")
    
    print("="*70 + "\n")

# ==========================================
# KHỐI LỆNH KÍCH HOẠT CHÍNH
# ==========================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLOv8 Inference')
    parser.add_argument('--source', type=str, default='test.jpg', help='Path to image or folder')
    parser.add_argument('--conf', type=float, default=CONFIDENCE_THRESHOLD, help='Confidence threshold')
    parser.add_argument('--output', type=str, default='./outputs', help='Output folder for results')
    
    args = parser.parse_args()
    
    # Tạo output folder nếu không tồn tại
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        model = load_model()
        source_path = Path(args.source)
        
        # Danh sách ảnh để xử lý
        image_paths = []
        if source_path.is_file():
            # Nếu là file
            image_paths = [source_path]
        elif source_path.is_dir():
            # Nếu là thư mục
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
            unique_paths = {}
            for ext in extensions:
                for p in source_path.glob(ext):
                    unique_paths[p.name] = p
            image_paths = list(unique_paths.values())
            print(f">>> Tìm thấy {len(image_paths)} ảnh không trùng lặp trong thư mục")
        else:
            print(f"❌ Path không tồn tại: {source_path}")
            exit(1)
        
        if not image_paths:
            print(f"❌ Không tìm thấy ảnh nào trong: {source_path}")
            exit(1)
        
        # Xử lý từng ảnh
        all_results = {}
        for idx, img_path in enumerate(image_paths, 1):
            print(f"\n[{idx}/{len(image_paths)}] Xử lý: {img_path.name}")
            result_json = run_inference(model, str(img_path), args.conf)
            all_results[img_path.name] = result_json
            
            # Vẽ và lưu ảnh với bounding box
            pred_img_path = draw_and_save_predictions(
                img_path, 
                result_json['detections'], 
                output_dir, 
                result_json['image_size']
            )
            if pred_img_path:
                print(f"   ✓ Ảnh: {pred_img_path}")
        
        # Lưu tổng kết quả
        summary_file = output_dir / 'batch_results.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)
        
        # In ra tóm tắt thống kê
        print_summary(all_results, args.conf)
        
        print(f"✓ Hoàn thành! Lưu kết quả vào: {output_dir}")
        print(f"   🖼 Ảnh có bounding box: {output_dir}/predictions/")
        print(f"   📄 Tệp tổng hợp: {summary_file}")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")