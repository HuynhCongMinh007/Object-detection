import os
import json
import time
import torch
import numpy as np
from collections import defaultdict
from inference import load_model, run_inference, load_class_names_from_coco, NUM_CLASSES

def calculate_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def plot_loss_curve(log_path, out_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Bỏ qua vẽ Loss Curve do chưa cài matplotlib.")
        return

    train_losses, val_losses = [], []
    if not os.path.exists(log_path): return
        
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if "Train Loss:" in line: train_losses.append(float(line.split(":")[1].strip()))
            elif "Val Loss:" in line: val_losses.append(float(line.split(":")[1].strip()))
            
    if not train_losses: return

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Valid Loss', marker='s')
    plt.title('Đường cong huấn luyện Loss (SSD)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(cm_dict, categories, out_dir):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Bỏ qua vẽ Confusion Matrix do chưa cài seaborn/matplotlib.")
        return

    names = sorted(list(categories)) + ['Background']
    mat = np.zeros((len(names), len(names)), dtype=int)
    name_to_idx = {name: i for i, name in enumerate(names)}
    
    for (gt, pd), count in cm_dict.items():
        mat[name_to_idx[gt], name_to_idx[pd]] = count
    mat[name_to_idx['Background'], name_to_idx['Background']] = 0

    plt.figure(figsize=(10, 8))
    sns.heatmap(mat, annot=True, fmt='d', cmap='Blues', xticklabels=names, yticklabels=names)
    plt.title('Ma trận nhầm lẫn - Confusion Matrix (SSD)')
    plt.xlabel('Nhãn Dự Đoán (Predicted)')
    plt.ylabel('Nhãn Thực Tế (Ground Truth)')
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_on_test_set(test_dir, model_path, log_path, iou_thresh=0.5, conf_thresh=0.5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ann_path = os.path.join(test_dir, '_annotations.coco.json')
    with open(ann_path, 'r', encoding='utf-8') as f: coco_data = json.load(f)

    category_map = {cat['id']: cat['name'] for cat in coco_data['categories']}
    idx_to_class = load_class_names_from_coco(ann_path)
    gt_boxes = defaultdict(list)
    for ann in coco_data['annotations']:
        cat_name = category_map[ann['category_id']]
        x1, y1, w, h = ann['bbox']
        gt_boxes[ann['image_id']].append({'bbox': [x1, y1, x1+w, y1+h], 'label': cat_name, 'matched': False})

    model = load_model(model_path, num_classes=NUM_CLASSES, device=device)
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    cm_dict = defaultdict(int)
    total_time, n_images = 0.0, 0
    
    for img_id, img_info in {img['id']: img for img in coco_data['images']}.items():
        img_path = os.path.join(test_dir, img_info['file_name'])
        if not os.path.exists(img_path): continue
        n_images += 1
        start_time = time.time()
        preds = run_inference(model, img_path, idx_to_class, confidence_threshold=conf_thresh, device=device)
        total_time += (time.time() - start_time)

        img_gts = gt_boxes.get(img_id, [])
        pred_dets = preds['detections']
        
        # --- 1. Đo theo tiêu chuẩn TP/FP/FN cho Precision/Recall ---
        for gt in img_gts: gt['matched'] = False
        for cat in category_map.values():
            c_preds = sorted([p for p in pred_dets if p['label'] == cat], key=lambda x: x['confidence'], reverse=True)
            c_gts = [g for g in img_gts if g['label'] == cat]
            for p in c_preds:
                best_iou, best_idx = 0.0, -1
                for idx, g in enumerate(c_gts):
                    if not g['matched']:
                        iou = calculate_iou(p['bbox'], g['bbox'])
                        if iou > best_iou: best_iou, best_idx = iou, idx
                if best_iou >= iou_thresh: c_gts[best_idx]['matched'] = True; tp[cat] += 1
                else: fp[cat] += 1
            fn[cat] += len([g for g in c_gts if not g['matched']])

        # --- 2. Đo Confusion Matrix (Bipartite Matching tự do) ---
        for gt in img_gts: gt['matched_cm'] = False
        for p in pred_dets:
            best_iou, best_idx = 0.0, -1
            for idx, g in enumerate(img_gts):
                if not g['matched_cm']:
                    iou = calculate_iou(p['bbox'], g['bbox'])
                    if iou > best_iou: best_iou, best_idx = iou, idx
            if best_iou >= iou_thresh:
                gt_label = img_gts[best_idx]['label']
                cm_dict[(gt_label, p['label'])] += 1
                img_gts[best_idx]['matched_cm'] = True
            else:
                cm_dict[('Background', p['label'])] += 1
        for g in img_gts:
            if not g['matched_cm']: cm_dict[(g['label'], 'Background')] += 1

    fps = n_images / total_time
    out_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    
    # Generate Images
    print("Vẽ biểu đồ Loss Curve và Confusion Matrix...")
    plot_loss_curve(log_path, out_dir)
    plot_confusion_matrix(cm_dict, category_map.values(), out_dir)

    # --- 3. Ghi file báo cáo Markdown ---
    report_lines = []
    report_lines.append("# KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH SSD\n")
    report_lines.append(f"- **Tổng số ảnh test:** {n_images}")
    report_lines.append(f"- **IoU Threshold:** {iou_thresh}")
    report_lines.append(f"- **Confidence Threshold:** {conf_thresh}")
    report_lines.append(f"- **Average FPS:** **{fps:.2f} frames/sec**\n")
    
    report_lines.append("### Hiệu năng chi tiết (Class-wise Metrics)\n")
    report_lines.append("| Lớp động vật | TP | FP | FN | Precision | Recall | F1-Score |")
    report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    p_list, r_list, f1_list = [], [], []
    for cat in sorted(category_map.values()):
        P = tp[cat] / (tp[cat]+fp[cat]) if (tp[cat]+fp[cat]) > 0 else 0
        R = tp[cat] / (tp[cat]+fn[cat]) if (tp[cat]+fn[cat]) > 0 else 0
        F1 = 2*P*R / (P+R) if (P+R) > 0 else 0
        p_list.append(P); r_list.append(R); f1_list.append(F1)
        report_lines.append(f"| **{cat}** | {tp[cat]} | {fp[cat]} | {fn[cat]} | {P*100:.2f}% | {R*100:.2f}% | {F1*100:.2f}% |")
        
    report_lines.append("\n### Chỉ số Trung bình (Mean Metrics)")
    report_lines.append(f"- **Mean Precision (mP):** **{np.mean(p_list)*100:.2f}%**")
    report_lines.append(f"- **Mean Recall (mR):** **{np.mean(r_list)*100:.2f}%**")
    report_lines.append(f"- **Mean F1-Score:** **{np.mean(f1_list)*100:.2f}%**")
    
    report_path = os.path.join(out_dir, 'evaluation_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"Đã lưu báo cáo tại: {report_path}")

if __name__ == '__main__':
    base = os.path.dirname(os.path.abspath(__file__))
    evaluate_on_test_set(
        os.path.join(base, 'animal_dataset', 'test'), 
        os.path.join(base, 'checkpoints', 'best_ssd.pth'),
        os.path.join(base, 'LogTrainModel.txt')
    )
