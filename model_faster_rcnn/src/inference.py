"""
inference.py - Bộ suy luận & Trực quan hóa cao cấp (Inference & Premium Visualization CLI).
Hỗ trợ chạy suy luận trên ảnh đơn lẻ, cả thư mục, đo lường độ trễ (latency), và vẽ bounding box bán trong suốt thời thượng.
"""
import os
import time
import argparse
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from config import *
from model import get_model
from nms import custom_nms


def predict_image(model, image_path, device,
                  score_threshold=SCORE_THRESHOLD,
                  nms_threshold=NMS_IOU_THRESHOLD):
    """
    Dự đoán bounding boxes trên một ảnh bất kỳ và đo lường thời gian xử lý.

    Quy trình:
    1. Đọc ảnh → chuyển sang tensor float [0, 1].
    2. Đưa qua mô hình ở chế độ eval → nhận predictions.
    3. Áp dụng custom_nms theo từng lớp (per-class NMS).
    4. Lọc các box có score >= ngưỡng.

    [Tham chiếu]: Đối chiếu luồng hoạt động này với Mục 4 trong 'inference_faster_rcnn.md'.

    Args:
        model: Mô hình Faster R-CNN đã huấn luyện.
        image_path (str): Đường dẫn tới ảnh cần dự đoán.
        device: torch.device (cuda hoặc cpu).
        score_threshold (float): Ngưỡng score tối thiểu.
        nms_threshold (float): Ngưỡng IoU cho NMS.

    Returns:
        image_np: Ảnh gốc numpy (để vẽ).
        final_boxes, final_labels, final_scores: Kết quả sau lọc.
        latency (float): Thời gian suy luận tính bằng mili-giây (ms).
    """
    # model.eval() chuyển mô hình sang trạng thái Suy luận (Mục 1, 2, 3, 4 trong inference_faster_rcnn.md).
    # Lúc này mô hình sẽ bỏ qua tính toán các Loss và trực tiếp lan truyền xuôi (Forward Pass).
    # RPN sẽ lọc tối đa 300 đề xuất (Post-NMS Top-N = 300) thay vì 2000 như lúc training.
    model.eval()

    # Đọc ảnh và chuyển sang tensor
    image_pil = Image.open(image_path).convert('RGB')
    image_np = np.array(image_pil)
    image_tensor = torch.as_tensor(
        image_np, dtype=torch.float32
    ).permute(2, 0, 1) / 255.0
    image_tensor = image_tensor.to(device)

    # Dự đoán & Đo lường thời gian suy luận (Inference Latency)
    start_time = time.time()
    with torch.no_grad():
        predictions = model([image_tensor])
    latency = (time.time() - start_time) * 1000  # Đơn vị: mili-giây (ms)

    pred = predictions[0]
    boxes = pred['boxes'].cpu()
    labels = pred['labels'].cpu()
    scores = pred['scores'].cpu()

    # === Áp dụng Hậu xử lý & Custom NMS theo từng lớp (Per-class NMS) ===
    # [Tham chiếu]: Đối chiếu toàn bộ bước hậu xử lý này với Mục 4.C trong 'inference_faster_rcnn.md'.
    all_boxes = []
    all_labels = []
    all_scores = []

    for class_id in range(1, NUM_CLASSES):  # Bước 1: Bỏ qua background (class 0)
        class_mask = labels == class_id
        if class_mask.sum() == 0:
            continue

        cls_boxes = boxes[class_mask]
        cls_scores = scores[class_mask]

        # Bước 3: Áp dụng custom NMS trên từng class riêng biệt để xóa hộp đè nhau
        keep = custom_nms(cls_boxes, cls_scores, nms_threshold)

        kept_boxes = cls_boxes[keep]
        kept_scores = cls_scores[keep]

        # Bước 1 & 2: Lọc theo ngưỡng tin cậy tối thiểu (confidence threshold)
        score_mask = kept_scores >= score_threshold
        all_boxes.append(kept_boxes[score_mask])
        all_labels.extend([class_id] * score_mask.sum().item())
        all_scores.append(kept_scores[score_mask])

    # Gộp kết quả từ tất cả các lớp
    if len(all_boxes) > 0:
        final_boxes = torch.cat(all_boxes, dim=0)
        final_scores = torch.cat(all_scores, dim=0)
        final_labels = torch.tensor(all_labels, dtype=torch.int64)
    else:
        final_boxes = torch.zeros((0, 4))
        final_scores = torch.zeros((0,))
        final_labels = torch.zeros((0,), dtype=torch.int64)

    return image_np, final_boxes, final_labels, final_scores, latency


def visualize_predictions(image_np, boxes, labels, scores, latency, save_path=None, show_image=True):
    """
    Trực quan hóa kết quả dự đoán với giao diện cao cấp:
    - Bounding Box nét căng, kết hợp dải nền bán trong suốt (alpha overlay) tạo hiệu ứng hiện đại.
    - Nhãn hiển thị bo tròn góc (rounded corners) sắc nét.
    - Hiển thị thông số FPS và Latency trực quan.

    Args:
        image_np: Ảnh numpy HWC uint8.
        boxes: Tensor [N, 4] bounding boxes.
        labels: Tensor [N] chỉ số lớp.
        scores: Tensor [N] điểm tin cậy.
        latency (float): Thời gian suy luận (ms).
        save_path: Đường dẫn lưu ảnh (tùy chọn).
        show_image (bool): Có hiển thị cửa sổ pyplot hay không.
    """
    fig, ax = plt.subplots(1, figsize=(14, 10))
    ax.imshow(image_np)

    num_objects = len(boxes)
    for i in range(num_objects):
        box = boxes[i]
        label_idx = labels[i].item()
        score = scores[i].item()

        xmin, ymin, xmax, ymax = box.tolist()
        w, h = xmax - xmin, ymax - ymin

        color = COLORS[label_idx] if label_idx < len(COLORS) else '#FFFFFF'
        cls_name = CLASS_NAMES[label_idx] if label_idx < len(CLASS_NAMES) else '?'

        # 1. Vẽ bounding box chính
        rect = patches.Rectangle(
            (xmin, ymin), w, h,
            linewidth=2.5, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)

        # 2. Vẽ dải nền bán trong suốt (alpha overlay) bên trong hộp - Tạo hiệu ứng vô cùng cao cấp
        rect_fill = patches.Rectangle(
            (xmin, ymin), w, h,
            linewidth=0, edgecolor='none', facecolor=color, alpha=0.15
        )
        ax.add_patch(rect_fill)

        # 3. Vẽ nhãn bo tròn nổi bật bên trên hộp
        ax.text(
            xmin, ymin - 6, f" {cls_name} {score*100:.1f}% ",
            fontsize=9, fontweight='bold', color='white',
            bbox=dict(
                boxstyle='round,pad=0.3', 
                facecolor=color, 
                edgecolor='none', 
                alpha=0.9,
                shadow=True
            ),
        )

    ax.set_axis_off()
    
    # Tiêu đề biểu diễn đầy đủ thông số
    fps = 1000.0 / latency if latency > 0 else 0.0
    ax.set_title(
        f"Faster R-CNN | Phát hiện: {num_objects} vật thể | "
        f"Độ trễ: {latency:.1f}ms | Tốc độ: {fps:.1f} FPS", 
        fontsize=14, fontweight='bold', color='#2C3E50', pad=15
    )
    plt.tight_layout()

    if save_path:
        # Tự động tạo thư mục cha nếu chưa có
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [Lưu ảnh]: Đã lưu kết quả trực quan hóa tại '{save_path}'")

    if show_image:
        plt.show()
    else:
        plt.close(fig)


def print_ascii_table(boxes, labels, scores, latency):
    """In bảng báo cáo kết quả đẹp mắt ngay trên Terminal."""
    print("\n" + "=" * 80)
    print(f"| BÁO CÁO PHÂN TÍCH HÌNH ẢNH (Độ trễ: {latency:.1f} ms | {1000.0/latency:.1f} FPS) ".ljust(79) + "|")
    print("=" * 80)
    print(f"| {'STT':<5} | {'Loài Vật (Class)':<20} | {'Độ Tin Cậy':<12} | {'Tọa độ Hộp (xmin, ymin, xmax, ymax)':<32} |")
    print("-" * 80)
    
    if len(boxes) == 0:
        print(f"| {'KPH':<5} | {'Không phát hiện thấy vật thể nào':<68} |")
    else:
        for idx in range(len(boxes)):
            cls = CLASS_NAMES[labels[idx].item()]
            scr = f"{scores[idx].item() * 100:.2f}%"
            coords = ", ".join([f"{int(c)}" for c in boxes[idx].tolist()])
            print(f"| {idx+1:<5} | {cls:<20} | {scr:<12} | {coords:<32} |")
            
    print("=" * 80 + "\n")


# ============================================================
# CLI INTERFACE
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Faster R-CNN Premium Inference Tool")
    parser.add_argument("--image", type=str, default=None, help="Đường dẫn đến 1 ảnh duy nhất cần dự đoán")
    parser.add_argument("--image_dir", type=str, default=None, help="Đường dẫn đến thư mục chứa danh sách ảnh")
    parser.add_argument("--demo", action="store_true", help="Chạy chế độ Demo ngẫu nhiên 5 ảnh trong tập test")
    parser.add_argument("--score_threshold", type=float, default=SCORE_THRESHOLD, help="Ngưỡng điểm tin cậy (0.0 - 1.0)")
    parser.add_argument("--nms_threshold", type=float, default=NMS_IOU_THRESHOLD, help="Ngưỡng khử trùng lặp IoU (0.0 - 1.0)")
    parser.add_argument("--model_path", type=str, default=os.path.join(OUTPUT_DIR, 'best_model.pth'), help="Đường dẫn file mô hình đã train (.pth)")
    parser.add_argument("--no_show", action="store_true", help="Không pop-up cửa sổ matplotlib (chỉ lưu ảnh kết quả)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Kiểm tra sự tồn tại của mô hình
    if not os.path.exists(args.model_path):
        print(f"\n[LỖI CẤU HÌNH]: Không tìm thấy file trọng số mô hình tại '{args.model_path}'.")
        print("Vui lòng thực hiện huấn luyện trước bằng cách chạy 'python train.py' trên Terminal/Colab.")
        return

    # 2. Tải mô hình
    print(f"\n=== ĐANG KHỞI TẠO MÔ HÌNH FASTER R-CNN ===")
    print(f"[*] Trọng số sử dụng: {args.model_path}")
    print(f"[*] Thiết bị tính toán: {DEVICE}")
    
    model = get_model(NUM_CLASSES)
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("[+] Khởi tạo thành công! Sẵn sàng suy luận.")

    # 3. Xác định nguồn ảnh đầu vào
    test_images = []
    
    # Trường hợp chạy 1 ảnh cụ thể
    if args.image:
        if os.path.exists(args.image):
            test_images = [args.image]
        else:
            print(f"[LỖI]: File ảnh '{args.image}' không tồn tại.")
            return
            
    # Trường hợp chạy cả thư mục
    elif args.image_dir:
        if os.path.exists(args.image_dir):
            all_files = os.listdir(args.image_dir)
            test_images = [os.path.join(args.image_dir, f) for f in all_files 
                           if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            print(f"[*] Tìm thấy {len(test_images)} ảnh trong thư mục '{args.image_dir}'")
        else:
            print(f"[LỖI]: Thư mục ảnh '{args.image_dir}' không tồn tại.")
            return
            
    # Trường hợp mặc định hoặc chọn --demo: Chạy trên 5 ảnh ngẫu nhiên của tập test
    else:
        test_dir = os.path.join(DATASET_DIR, 'test')
        if os.path.exists(test_dir):
            all_test_files = [f for f in os.listdir(test_dir)
                              if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            num_demo = min(5, len(all_test_files))
            selected = np.random.choice(all_test_files, num_demo, replace=False)
            test_images = [os.path.join(test_dir, f) for f in selected]
            print(f"[*] Chế độ Demo: Chọn ngẫu nhiên {num_demo} ảnh từ tập kiểm tra '{test_dir}'")
        else:
            print(f"[LỖI]: Không tìm thấy thư mục tập test cục bộ tại '{test_dir}'.")
            print("Vui lòng truyền đường dẫn ảnh thủ công bằng tham số '--image đường_dẫn_ảnh'.")
            return

    # 4. Chạy suy luận và trực quan hóa
    for idx, img_path in enumerate(test_images):
        img_name = os.path.basename(img_path)
        print(f"\n[{idx+1}/{len(test_images)}] Đang xử lý: {img_name}")
        
        image_np, boxes, labels, scores = predict_image(
            model, img_path, DEVICE
        )

        print(f"  Detected {len(boxes)} objects:")
        for i in range(len(boxes)):
            cls = CLASS_NAMES[labels[i].item()]
            scr = scores[i].item()
            print(f"    [{cls}] score={scr:.3f}")

        save_name = os.path.join(OUTPUT_DIR, f"result_{img_name}")
        visualize_predictions(
            image_np, boxes, labels, scores, latency,
            save_path=save_name,
            show_image=not args.no_show
        )

    print("\n" + "=" * 80)
    print("=== TẤT CẢ KẾT QUẢ ĐÃ ĐƯỢC SUY LUẬN XONG ===")
    print(f"[*] Ảnh kết quả được lưu tại thư mục: '{OUTPUT_DIR}/'")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
