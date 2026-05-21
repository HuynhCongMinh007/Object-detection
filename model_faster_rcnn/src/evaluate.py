"""
evaluate.py - Đánh giá chuyên sâu mô hình Faster R-CNN (Error Analysis & Metrics).
Đo lường:
- Precision, Recall, F1-Score cho từng lớp vật thể (áp dụng IoU matching).
- Ma trận nhầm lẫn phát hiện vật thể (Object Detection Confusion Matrix) gồm cả lớp Nền (Background).
- Tìm ra Top 3 các cặp loài vật mô hình dễ bị nhầm lẫn nhất (Top 3 Confused Classes).
- Tự động lưu biểu đồ Confusion Matrix Heatmap và Báo cáo đánh giá chi tiết (evaluation_report.md).
"""
import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

from config import *
from dataset import AnimalDataset, get_valid_transforms, collate_fn
from model import get_model
from nms import custom_nms


def compute_iou(box1, box2):
    """Tính toán IoU giữa hai Bounding Box."""
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    
    inter_xmin = max(xmin1, xmin2)
    inter_ymin = max(ymin1, ymin2)
    inter_xmax = min(xmax1, xmax2)
    inter_ymax = min(ymax1, ymax2)
    
    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h
    
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    return inter_area / union_area


@torch.no_grad()
def evaluate_dataset(model, data_loader, device, score_threshold=0.5, iou_threshold=0.5):
    """
    Duyệt toàn bộ tập dữ liệu để đánh giá định lượng mô hình.
    
    Quy trình đối sánh (IoU Box Matching):
    - Chạy mô hình trên từng ảnh để nhận các dự đoán.
    - Dùng custom NMS để loại bỏ box trùng lặp.
    - Với mỗi Box dự đoán, tìm Box Ground Truth trùng khớp nhất (IoU >= 0.5).
    - Cập nhật ma trận nhầm lẫn kích thước [7, 7] (gồm cả background).
    """
    model.eval()
    
    # Ma trận nhầm lẫn: 7 lớp (0: Nền/Background, 1-6: Động vật)
    # cm[true_class][pred_class]
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    
    print("[*] Đang chạy đánh giá định lượng trên toàn bộ tập Validation...")
    
    for idx, (images, targets) in enumerate(data_loader):
        # Forward pass
        images_device = [img.to(device) for img in images]
        predictions = model(images_device)
        
        for i in range(len(images)):
            pred = predictions[i]
            target = targets[i]
            
            gt_boxes = target['boxes'].numpy()
            gt_labels = target['labels'].numpy()
            
            p_boxes_raw = pred['boxes'].cpu()
            p_labels_raw = pred['labels'].cpu()
            p_scores_raw = pred['scores'].cpu()
            
            # Áp dụng Custom Per-class NMS
            keep_indices = []
            for class_id in range(1, NUM_CLASSES):
                class_mask = p_labels_raw == class_id
                if class_mask.sum() == 0:
                    continue
                
                cls_boxes = p_boxes_raw[class_mask]
                cls_scores = p_scores_raw[class_mask]
                
                keep = custom_nms(cls_boxes, cls_scores, NMS_IOU_THRESHOLD)
                
                # Ánh xạ chỉ số keep về mảng gốc
                orig_indices = torch.where(class_mask)[0]
                keep_indices.extend(orig_indices[keep].tolist())
                
            if len(keep_indices) > 0:
                p_boxes = p_boxes_raw[keep_indices].numpy()
                p_labels = p_labels_raw[keep_indices].numpy()
                p_scores = p_scores_raw[keep_indices].numpy()
            else:
                p_boxes = np.zeros((0, 4))
                p_labels = np.array([])
                p_scores = np.array([])
                
            # Lọc theo score threshold
            score_mask = p_scores >= score_threshold
            p_boxes = p_boxes[score_mask]
            p_labels = p_labels[score_mask]
            
            # Khởi tạo mảng đánh dấu đã so khớp
            gt_matched = np.zeros(len(gt_labels), dtype=bool)
            pred_matched = np.zeros(len(p_labels), dtype=bool)
            
            # 1. So khớp các dự đoán có IoU cao với Ground Truth (cùng class)
            for p_idx in range(len(p_labels)):
                p_box = p_boxes[p_idx]
                p_label = p_labels[p_idx]
                
                best_iou = 0.0
                best_gt_idx = -1
                
                for gt_idx in range(len(gt_labels)):
                    if gt_matched[gt_idx] or gt_labels[gt_idx] != p_label:
                        continue
                    
                    iou = compute_iou(p_box, gt_boxes[gt_idx])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                        
                if best_iou >= iou_threshold:
                    # Đúng hoàn toàn (True Positive)
                    gt_matched[best_gt_idx] = True
                    pred_matched[p_idx] = True
                    cm[p_label][p_label] += 1
            
            # 2. Phát hiện các lỗi phân loại nhầm loài vật (Class confusion)
            for p_idx in range(len(p_labels)):
                if pred_matched[p_idx]:
                    continue
                    
                p_box = p_boxes[p_idx]
                p_label = p_labels[p_idx]
                
                best_iou = 0.0
                best_gt_idx = -1
                
                for gt_idx in range(len(gt_labels)):
                    if gt_matched[gt_idx]:
                        continue
                    
                    iou = compute_iou(p_box, gt_boxes[gt_idx])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                        
                if best_iou >= iou_threshold:
                    # Mô hình xác định đúng có vật thể nhưng dự đoán sai nhãn
                    gt_matched[best_gt_idx] = True
                    pred_matched[p_idx] = True
                    true_label = gt_labels[best_gt_idx]
                    cm[true_label][p_label] += 1  # Nhầm lẫn true_label thành p_label
            
            # 3. Phát hiện lỗi nhận diện nhầm nền thành vật thể (False Positives - Hallucinations)
            for p_idx in range(len(p_labels)):
                if not pred_matched[p_idx]:
                    p_label = p_labels[p_idx]
                    cm[0][p_label] += 1  # True là Background (0) nhưng đoán là p_label
                    
            # 4. Phát hiện lỗi bỏ sót vật thể (False Negatives - Misses)
            for gt_idx in range(len(gt_labels)):
                if not gt_matched[gt_idx]:
                    true_label = gt_labels[gt_idx]
                    cm[true_label][0] += 1  # True là true_label nhưng đoán là Background (0)
                    
        if (idx + 1) % 10 == 0 or (idx + 1) == len(data_loader):
            print(f"  Processed [{idx+1}/{len(data_loader)}] batches...")
            
    return cm


def calculate_metrics(cm):
    """Tính toán Precision, Recall, F1 cho từng lớp động vật (bỏ qua background)."""
    metrics = {}
    for idx in range(1, NUM_CLASSES):
        cls_name = CLASS_NAMES[idx]
        
        tp = cm[idx][idx]
        fp = cm[0][idx] + sum(cm[other][idx] for other in range(1, NUM_CLASSES) if other != idx)
        fn = cm[idx][0] + sum(cm[idx][other] for other in range(1, NUM_CLASSES) if other != idx)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[cls_name] = {
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    return metrics


def plot_and_save_confusion_matrix(cm):
    """Vẽ heatmap ma trận nhầm lẫn nâng cao sử dụng seaborn."""
    plt.figure(figsize=(11, 9))
    
    # Chuẩn hóa ma trận nhầm lẫn theo dòng (True Labels) để hiển thị phần trăm tỉ lệ
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1 # Tránh chia cho 0
    cm_percent = (cm / row_sums) * 100.0
    
    # Tạo các nhãn annotation chứa cả số lượng tuyệt đối và tỷ lệ phần trăm
    labels_anno = np.empty_like(cm, dtype=object)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            labels_anno[i, j] = f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)"
            
    sns.heatmap(
        cm_percent, 
        annot=labels_anno, 
        fmt='', 
        cmap='Blues', 
        xticklabels=CLASS_NAMES, 
        yticklabels=CLASS_NAMES,
        cbar=True,
        linewidths=0.5
    )
    
    plt.title('Ma Trận Nhầm Lẫn Đối Sánh Phát Hiện Vật Thể\n(Object Detection Confusion Matrix)', fontsize=13, fontweight='bold', pad=15)
    plt.ylabel('Nhãn Thực Tế (True Label)', fontsize=11, fontweight='bold')
    plt.xlabel('Nhãn Dự Đoán (Predicted Label)', fontsize=11, fontweight='bold')
    plt.tight_layout()
    
    cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[Đồ thị]: Đã lưu Heatmap Ma trận nhầm lẫn tại '{cm_path}'")


def analyze_top_confusions(cm):
    """Tìm ra 3 cặp lớp dễ bị mô hình phân loại nhầm lẫn nhất (bỏ qua background)."""
    confusions = []
    for i in range(1, NUM_CLASSES):
        for j in range(1, NUM_CLASSES):
            if i == j:
                continue
            count = cm[i][j]
            if count > 0:
                confusions.append((CLASS_NAMES[i], CLASS_NAMES[j], int(count)))
                
    # Sắp xếp giảm dần theo số lượng lỗi nhầm lẫn
    confusions.sort(key=lambda x: x[2], reverse=True)
    return confusions[:3]


def generate_evaluation_report(metrics, top_confusions, cm):
    """Tự động xuất file báo cáo đánh giá định lượng chuyên sâu evaluation_report.md."""
    report_path = os.path.join(OUTPUT_DIR, 'evaluation_report.md')
    
    # Tính toán mAP@0.5 (Mean F1/Mean Precision/Mean Recall)
    mean_prec = np.mean([m['precision'] for m in metrics.values()])
    mean_rec = np.mean([m['recall'] for m in metrics.values()])
    mean_f1 = np.mean([m['f1_score'] for m in metrics.values()])
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# BÁO CÁO ĐÁNH GIÁ ĐỊNH LƯỢNG VÀ PHÂN TÍCH LỖI (ERROR ANALYSIS)\n\n")
        
        f.write("## 1. Chỉ số đánh giá chung cuộc trên tập Validation\n")
        f.write(f"- **Mean Precision (mPrecision):** {mean_prec*100:.2f}%\n")
        f.write(f"- **Mean Recall (mRecall):** {mean_rec*100:.2f}%\n")
        f.write(f"- **Mean F1-Score (mF1):** {mean_f1*100:.2f}%\n")
        f.write("- **Điều kiện đối khớp (Matching Criteria):** IoU >= 0.50 & Score >= 0.50\n\n")
        
        f.write("## 2. Báo cáo Chi tiết từng Lớp Loài vật (Class-wise Performance)\n")
        f.write("| Loài vật (Class) | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Precision | Recall | F1-Score |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        
        for cls_name, m in metrics.items():
            f.write(f"| {cls_name} | {m['tp']} | {m['fp']} | {m['fn']} | {m['precision']*100:.2f}% | {m['recall']*100:.2f}% | {m['f1_score']*100:.2f}% |\n")
            
        f.write("\n## 3. Phân tích Lỗi Nhầm lẫn Hàng đầu (Top 3 Confusions)\n")
        if len(top_confusions) == 0:
            f.write("- Tuyệt vời! Mô hình không có lỗi nhầm lẫn nhãn chéo giữa các loài vật.\n")
        else:
            f.write("Dưới đây là 3 cặp loài vật mà mô hình dễ nhận diện nhầm lẫn cho nhau nhất:\n")
            for idx, (true_cls, pred_cls, count) in enumerate(top_confusions):
                f.write(f"{idx+1}. **Thực tế:** `{true_cls}` nhưng mô hình nhận nhầm thành `{pred_cls}`: **{count} lần**\n")
                
        f.write("\n## 4. Phân tích Lỗi Bỏ sót (False Negatives) & Nhận nhầm Nền (False Positives)\n")
        f.write("- **Lỗi Bỏ Sót vật thể (Missed Objects):**\n")
        for idx in range(1, NUM_CLASSES):
            cls_name = CLASS_NAMES[idx]
            missed = cm[idx][0]
            f.write(f"  + Lớp `{cls_name}` bị bỏ sót hoàn toàn (dự đoán là Nền): **{missed} lần**\n")
            
        f.write("\n- **Lỗi Nhận diện sai Nền (Hallucinations/False Alarms):**\n")
        for idx in range(1, NUM_CLASSES):
            cls_name = CLASS_NAMES[idx]
            false_alarm = cm[0][idx]
            f.write(f"  + Vùng Nền trống bị mô hình dự đoán nhầm thành `{cls_name}`: **{false_alarm} lần**\n")
            
        f.write("\n## 5. Heatmap Ma trận nhầm lẫn trực quan\n")
        f.write("![Confusion Matrix Heatmap](confusion_matrix.png)\n")
        
    print(f"[Báo cáo]: Đã sinh báo cáo phân tích lỗi định lượng tại '{report_path}'")


def main():
    model_path = os.path.join(OUTPUT_DIR, 'best_model.pth')
    if not os.path.exists(model_path):
        print(f"[LỖI]: Không tìm thấy mô hình đã huấn luyện tại '{model_path}'. Vui lòng train trước!")
        return

    print("=" * 60)
    print("BẮT ĐẦU ĐÁNH GIÁ CHUYÊN SÂU FASTER R-CNN (BÀI LAB 3 EXTENDED)")
    print("=" * 60)

    # 1. Khởi tạo dữ liệu
    valid_dataset = AnimalDataset(
        os.path.join(DATASET_DIR, 'valid'), transforms=get_valid_transforms()
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True
    )
    
    # 2. Khởi tạo mô hình
    model = get_model(NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # 3. Đánh giá tạo ma trận nhầm lẫn
    cm = evaluate_dataset(model, valid_loader, DEVICE)

    # 4. Tính toán số liệu hiệu năng
    metrics = calculate_metrics(cm)

    # 5. Phân tích 3 lỗi nhầm lẫn lớn nhất
    top_confusions = analyze_top_confusions(cm)

    # In kết quả nhanh trên console
    print("\n" + "=" * 60)
    print("KẾT QUẢ ĐÁNH GIÁ NHANH TRÊN CONSOLE:")
    print("=" * 60)
    for cls_name, m in metrics.items():
        print(f"Class [{cls_name:<10}] | Precision: {m['precision']*100:.1f}% | Recall: {m['recall']*100:.1f}% | F1: {m['f1_score']*100:.1f}%")
    print("=" * 60)
    
    print("\n[Top 3 lỗi nhầm lẫn chéo]:")
    for idx, (t_cls, p_cls, count) in enumerate(top_confusions):
        print(f"  {idx+1}. Thực tế: {t_cls:<10} -> Nhầm thành: {p_cls:<10} | Số lần: {count}")
    print("=" * 60 + "\n")

    # 6. Trực quan hóa và lưu Heatmap Confusion Matrix
    plot_and_save_confusion_matrix(cm)

    # 7. Tự động xuất báo cáo đánh giá định lượng chuyên sâu Markdown
    generate_evaluation_report(metrics, top_confusions, cm)
    
    print("=== TIẾN TRÌNH ĐÁNH GIÁ HOÀN TẤT THÀNH CÔNG ===")


if __name__ == '__main__':
    main()
