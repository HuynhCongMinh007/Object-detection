import os
import cv2
import numpy as np
import random
import glob
import argparse
import sys

# Khắc phục lỗi UnicodeEncodeError trên Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


class YOLOAugmenter:
    def __init__(self, data_dir='./data', classes=None):
        self.data_dir = data_dir
        self.train_images_dir = os.path.join(data_dir, 'train', 'images')
        self.train_labels_dir = os.path.join(data_dir, 'train', 'labels')
        # Danh sách các lớp mặc định từ data.yaml
        self.classes = classes or ['cat', 'chicken', 'cow', 'dog', 'horse', 'sheep']

    def load_yolo_labels(self, label_path):
        """Đọc file nhãn YOLO và trả về danh sách [class_id, coords] (coords có thể là box hoặc polygon)"""
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    class_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    labels.append((class_id, coords))
        return labels

    def save_yolo_labels(self, label_path, labels):
        """Lưu danh sách nhãn YOLO xuống file .txt"""
        with open(label_path, 'w') as f:
            for class_id, coords in labels:
                coord_str = " ".join([f"{x:.6f}" for x in coords])
                f.write(f"{class_id} {coord_str}\n")

    def clean_augmented_files(self):
        """Xóa tất cả các ảnh và nhãn đã được tăng cường thủ công trước đó"""
        print("🧹 Đang dọn dẹp các ảnh và nhãn tăng cường cũ (có hậu tố '_manual_aug_')...")
        img_patterns = [
            os.path.join(self.train_images_dir, '*_manual_aug_*')
        ]
        label_patterns = [
            os.path.join(self.train_labels_dir, '*_manual_aug_*')
        ]
        
        deleted_imgs = 0
        for pattern in img_patterns:
            for f in glob.glob(pattern):
                try:
                    os.remove(f)
                    deleted_imgs += 1
                except Exception as e:
                    print(f"❌ Lỗi khi xóa file ảnh {f}: {e}")
                    
        deleted_labels = 0
        for pattern in label_patterns:
            for f in glob.glob(pattern):
                try:
                    os.remove(f)
                    deleted_labels += 1
                except Exception as e:
                    print(f"❌ Lỗi khi xóa file nhãn {f}: {e}")
                    
        print(f"🗑️ Đã dọn dẹp: xóa thành công {deleted_imgs} ảnh và {deleted_labels} file nhãn tăng cường.")
        return deleted_imgs

    def augment_image_and_labels(self, img, labels):
        """
        Áp dụng các phép biến đổi ngẫu nhiên lên ảnh và tính toán lại bounding box hoặc polygon tương ứng.
        """
        augmented_img = img.copy()
        augmented_labels = [list(x) for x in labels]  # Sao chép nhãn tránh sửa đổi ảnh gốc
        H, W = img.shape[:2]

        # 1. Lật ngang ngẫu nhiên (Tỉ lệ 50%)
        if random.random() < 0.5:
            augmented_img = cv2.flip(augmented_img, 1)
            for item in augmented_labels:
                coords = item[1]
                if len(coords) == 4:
                    # Định dạng Bounding Box: [xc, yc, w, h]
                    coords[0] = 1.0 - coords[0]  # x_center mới = 1.0 - x_center cũ
                else:
                    # Định dạng Polygon: [x1, y1, x2, y2, ...]
                    for i in range(0, len(coords), 2):
                        coords[i] = 1.0 - coords[i]

        # 2. Thay đổi ngẫu nhiên Độ sáng & Độ tương phản (Tỉ lệ 70%)
        if random.random() < 0.7:
            alpha = random.uniform(0.8, 1.2)   # Độ tương phản
            beta = random.randint(-25, 25)     # Độ sáng
            augmented_img = cv2.convertScaleAbs(augmented_img, alpha=alpha, beta=beta)

        # 3. Làm mờ ảnh ngẫu nhiên (Tỉ lệ 40%)
        if random.random() < 0.4:
            ksize = random.choice([3, 5])
            augmented_img = cv2.GaussianBlur(augmented_img, (ksize, ksize), 0)

        # 4. Thêm nhiễu Gaussian ngẫu nhiên (Tỉ lệ 35%)
        if random.random() < 0.35:
            mean = 0
            sigma = random.uniform(5, 15)
            gauss = np.random.normal(mean, sigma, augmented_img.shape).astype(np.float32)
            augmented_img = np.clip(augmented_img.astype(np.float32) + gauss, 0, 255).astype(np.uint8)

        # 5. Dịch chuyển, Thu phóng và Xoay nhẹ (Affine Transform) (Tỉ lệ 60%)
        if random.random() < 0.6:
            angle = random.uniform(-10, 10)       # Xoay ngẫu nhiên từ -10 đến 10 độ
            scale = random.uniform(0.85, 1.15)    # Thu phóng ngẫu nhiên từ 0.85 đến 1.15
            tx = random.uniform(-0.06, 0.06) * W  # Dịch chuyển trục X (tối đa 6% ảnh)
            ty = random.uniform(-0.06, 0.06) * H  # Dịch chuyển trục Y (tối đa 6% ảnh)
            
            # Khởi tạo ma trận Affine xoay quanh tâm ảnh
            cx, cy = W / 2, H / 2
            M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
            # Thêm độ dịch chuyển tX, tY vào ma trận xoay
            M[0, 2] += tx
            M[1, 2] += ty
            
            # Thực hiện xoay/dịch ảnh, bù viền bằng màu xám trung tính (128, 128, 128) để YOLO không bị nhiễu
            augmented_img = cv2.warpAffine(
                augmented_img, M, (W, H), 
                borderMode=cv2.BORDER_CONSTANT, 
                borderValue=(128, 128, 128)
            )
            
            # Tính toán lại tọa độ sau phép biến đổi Affine
            new_labels = []
            for class_id, coords in augmented_labels:
                if len(coords) == 4:
                    # Bounding Box format: [xc, yc, w, h]
                    xc, yc, w, h = coords
                    # 4 đỉnh của bounding box cũ (tọa độ pixel)
                    x1, y1 = (xc - w/2) * W, (yc - h/2) * H
                    x2, y2 = (xc + w/2) * W, (yc - h/2) * H
                    x3, y3 = (xc - w/2) * W, (yc + h/2) * H
                    x4, y4 = (xc + w/2) * W, (yc + h/2) * H
                    
                    corners = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                    ones = np.ones((4, 1))
                    corners_hom = np.hstack((corners, ones))  # ma trận thuần nhất 4x3
                    transformed_corners = M.dot(corners_hom.T).T  # 4x2
                    
                    # Tìm tọa độ x_min, y_min, x_max, y_max mới
                    tx_coords = transformed_corners[:, 0]
                    ty_coords = transformed_corners[:, 1]
                    
                    new_x1 = np.clip(np.min(tx_coords), 0, W)
                    new_y1 = np.clip(np.min(ty_coords), 0, H)
                    new_x2 = np.clip(np.max(tx_coords), 0, W)
                    new_y2 = np.clip(np.max(ty_coords), 0, H)
                    
                    new_w_px = new_x2 - new_x1
                    new_h_px = new_y2 - new_y1
                    
                    # Chỉ giữ lại nhãn nếu nó còn hiển thị trong ảnh (chiều ngang & dọc >= 8 pixel)
                    if new_w_px >= 8 and new_h_px >= 8:
                        new_xc = (new_x1 + new_w_px / 2.0) / W
                        new_yc = (new_y1 + new_h_px / 2.0) / H
                        new_w = new_w_px / W
                        new_h = new_h_px / H
                        new_labels.append((class_id, [new_xc, new_yc, new_w, new_h]))
                else:
                    # Polygon format: [x1, y1, x2, y2, ...]
                    pts = np.array(coords).reshape(-1, 2)
                    pts_px = pts * [W, H]
                    
                    ones = np.ones((len(pts_px), 1))
                    pts_hom = np.hstack((pts_px, ones))
                    transformed_pts = M.dot(pts_hom.T).T # (N, 2)
                    
                    # Giới hạn điểm trong vùng ảnh
                    transformed_pts[:, 0] = np.clip(transformed_pts[:, 0], 0, W)
                    transformed_pts[:, 1] = np.clip(transformed_pts[:, 1], 0, H)
                    
                    # Kiểm tra xem đa giác sau biến đổi có còn nhìn thấy rõ không thông qua bbox của nó
                    tx_coords = transformed_pts[:, 0]
                    ty_coords = transformed_pts[:, 1]
                    new_w_px = np.max(tx_coords) - np.min(tx_coords)
                    new_h_px = np.max(ty_coords) - np.min(ty_coords)
                    
                    if new_w_px >= 8 and new_h_px >= 8:
                        transformed_pts_norm = transformed_pts / [W, H]
                        new_coords = transformed_pts_norm.flatten().tolist()
                        new_labels.append((class_id, new_coords))
            
            augmented_labels = new_labels

        return augmented_img, augmented_labels

    def generate_augmentations(self, target_count=4000):
        """
        Thực hiện quét ảnh gốc và nhân bản thêm ảnh tăng cường cho đến khi tập train đạt target_count ảnh.
        """
        # Quét các ảnh gốc (không có hậu tố '_manual_aug_')
        original_img_paths = [
            p for p in glob.glob(os.path.join(self.train_images_dir, '*'))
            if os.path.isfile(p) and '_manual_aug_' not in p and p.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        
        orig_count = len(original_img_paths)
        if orig_count == 0:
            print(f"❌ Lỗi: Không tìm thấy ảnh gốc nào trong {self.train_images_dir}!")
            return
            
        current_total = len([p for p in glob.glob(os.path.join(self.train_images_dir, '*')) 
                             if os.path.isfile(p) and p.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        print(f"📊 Số lượng ảnh gốc phát hiện: {orig_count}")
        print(f"📊 Số lượng ảnh hiện có trong tập train: {current_total}")
        
        if current_total >= target_count:
            print(f"✅ Số lượng ảnh hiện tại ({current_total}) đã đạt hoặc vượt mức mục tiêu ({target_count}).")
            print("   -> Bỏ qua quá trình sinh ảnh tăng cường. Nếu muốn chạy lại từ đầu, vui lòng thêm tham số --clean.")
            return

        needed = target_count - current_total
        print(f"🚀 Bắt đầu sinh thêm {needed} ảnh tăng cường...")
        
        generated = 0
        failed_attempts = 0
        
        while generated < needed:
            # Chọn ngẫu nhiên một ảnh gốc
            orig_img_path = random.choice(original_img_paths)
            base = os.path.basename(orig_img_path)
            name, ext = os.path.splitext(base)
            
            # Đọc ảnh gốc
            img = cv2.imread(orig_img_path)
            if img is None:
                failed_attempts += 1
                if failed_attempts > 100:
                    print("❌ Lỗi liên tục khi đọc ảnh gốc. Dừng tiến trình.")
                    break
                continue
                
            # Đọc nhãn tương ứng
            label_path = os.path.join(self.train_labels_dir, name + '.txt')
            labels = self.load_yolo_labels(label_path)
            
            # Áp dụng tăng cường dữ liệu
            aug_img, aug_labels = self.augment_image_and_labels(img, labels)
            
            # Nếu ảnh gốc có nhãn nhưng ảnh tăng cường bị mất sạch nhãn do dịch chuyển/cắt quá đà -> sinh lại
            if len(labels) > 0 and len(aug_labels) == 0:
                continue
                
            # Tạo tên file mới với hậu tố '_manual_aug_{index}'
            aug_name = f"{name}_manual_aug_{generated}"
            aug_img_path = os.path.join(self.train_images_dir, aug_name + ext)
            aug_label_path = os.path.join(self.train_labels_dir, aug_name + '.txt')
            
            # Lưu ảnh và nhãn đã biến đổi
            cv2.imwrite(aug_img_path, aug_img)
            self.save_yolo_labels(aug_label_path, aug_labels)
            
            generated += 1
            if generated % 100 == 0 or generated == needed:
                print(f"   -> Tiến độ: Đã sinh {generated}/{needed} ảnh...")
                
        # Thống kê sau khi hoàn thành
        final_total = len([p for p in glob.glob(os.path.join(self.train_images_dir, '*')) 
                           if os.path.isfile(p) and p.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"🎉 Hoàn tất! Đã sinh {generated} ảnh tăng cường thành công.")
        print(f"📊 Tổng số lượng ảnh tập train hiện tại: {final_total}")

    def draw_bboxes(self, img, labels):
        """Vẽ bounding box hoặc polygon của nhãn YOLO lên ảnh để trực quan hóa"""
        H, W = img.shape[:2]
        viz_img = img.copy()
        colors = [
            (255, 0, 0),     # Xanh dương - cat
            (0, 255, 0),     # Xanh lá - chicken
            (0, 0, 255),     # Đỏ - cow
            (255, 255, 0),   # Xanh cyan - dog
            (255, 0, 255),   # Hồng cánh sen - horse
            (0, 255, 255)    # Vàng - sheep
        ]
        for class_id, coords in labels:
            color = colors[class_id % len(colors)]
            class_name = self.classes[class_id] if class_id < len(self.classes) else f"class_{class_id}"
            
            if len(coords) == 4:
                # Bounding box format
                xc, yc, w, h = coords
                x1 = int((xc - w/2) * W)
                y1 = int((yc - h/2) * H)
                x2 = int((xc + w/2) * W)
                y2 = int((yc + h/2) * H)
                
                # Vẽ hình chữ nhật bounding box
                cv2.rectangle(viz_img, (x1, y1), (x2, y2), color, 2)
                
                # Vẽ nền chữ nhãn
                label_text = f"{class_name}"
                (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(viz_img, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
                cv2.putText(viz_img, label_text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            else:
                # Polygon format
                pts = np.array(coords).reshape(-1, 2)
                pts_px = (pts * [W, H]).astype(np.int32)
                
                # Vẽ đa giác
                cv2.polylines(viz_img, [pts_px], isClosed=True, color=color, thickness=2)
                
                # Vẽ nhãn tại điểm trên cùng bên trái của đa giác
                x1 = int(np.min(pts_px[:, 0]))
                y1 = int(np.min(pts_px[:, 1]))
                
                label_text = f"{class_name}"
                (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(viz_img, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
                cv2.putText(viz_img, label_text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
                
        return viz_img

    def save_visualizations(self, num_samples=10):
        """Xuất ảnh mẫu so sánh giữa ảnh gốc và ảnh tăng cường vào outputs/augmented_samples"""
        viz_dir = os.path.join(os.path.dirname(self.train_images_dir), '..', '..', 'outputs', 'augmented_samples')
        viz_dir = os.path.abspath(viz_dir)
        os.makedirs(viz_dir, exist_ok=True)
        
        # Dọn dẹp các mẫu trực quan hóa cũ
        for f in glob.glob(os.path.join(viz_dir, '*')):
            try:
                os.remove(f)
            except:
                pass
                
        # Tìm các ảnh gốc
        img_paths = [p for p in glob.glob(os.path.join(self.train_images_dir, '*')) 
                     if os.path.isfile(p) and '_manual_aug_' not in p and p.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not img_paths:
            print("⚠️ Không tìm thấy ảnh gốc để xuất mẫu trực quan hóa.")
            return
            
        samples = random.sample(img_paths, min(num_samples, len(img_paths)))
        print(f"🖼️ Đang xuất {len(samples)} mẫu đối chiếu ảnh gốc và ảnh đã tăng cường vào: {viz_dir}")
        
        for idx, img_path in enumerate(samples):
            base = os.path.basename(img_path)
            name, ext = os.path.splitext(base)
            
            # Đọc ảnh gốc và vẽ bounding box
            img = cv2.imread(img_path)
            label_path = os.path.join(self.train_labels_dir, name + '.txt')
            labels = self.load_yolo_labels(label_path)
            orig_viz = self.draw_bboxes(img, labels)
            
            # Tìm ảnh tăng cường tương ứng
            aug_pattern = os.path.join(self.train_images_dir, f"{name}_manual_aug_*{ext}")
            aug_paths = glob.glob(aug_pattern)
            
            if aug_paths:
                aug_path = aug_paths[0]
                aug_img = cv2.imread(aug_path)
                aug_name = os.path.splitext(os.path.basename(aug_path))[0]
                aug_label_path = os.path.join(self.train_labels_dir, aug_name + '.txt')
                aug_labels = self.load_yolo_labels(aug_label_path)
                aug_viz = self.draw_bboxes(aug_img, aug_labels)
                
                # Ghép đôi hai ảnh ngang hàng để so sánh trực quan
                if img.shape[0] == aug_img.shape[0] and img.shape[1] == aug_img.shape[1]:
                    combined = np.hstack((orig_viz, aug_viz))
                    # Thêm phân cách giữa 2 ảnh bằng đường kẻ đỏ dọc
                    cv2.line(combined, (img.shape[1], 0), (img.shape[1], img.shape[0]), (0, 0, 255), 3)
                    cv2.imwrite(os.path.join(viz_dir, f"compare_{idx}_{name}.jpg"), combined)
                else:
                    cv2.imwrite(os.path.join(viz_dir, f"compare_{idx}_{name}_orig.jpg"), orig_viz)
                    cv2.imwrite(os.path.join(viz_dir, f"compare_{idx}_{name}_aug.jpg"), aug_viz)
            else:
                cv2.imwrite(os.path.join(viz_dir, f"compare_{idx}_{name}_orig.jpg"), orig_viz)
                
        print("✅ Đã tạo xong ảnh mẫu so sánh trực quan!")

def main():
    parser = argparse.ArgumentParser(description="Tăng cường dữ liệu YOLOv8 thủ công và offline")
    parser.add_argument('--data-dir', type=str, default='./data', help='Thư mục chứa data')
    parser.add_argument('--target-count', type=int, default=4000, help='Tổng số lượng ảnh train mục tiêu sau tăng cường')
    parser.add_argument('--clean', action='store_true', help='Dọn dẹp tất cả ảnh tăng cường cũ trước khi chạy mới')
    parser.add_argument('--no-viz', action='store_true', help='Không xuất mẫu ảnh so sánh nhãn trực quan')
    
    args = parser.parse_args()
    
    # Khởi tạo đối tượng tăng cường
    augmenter = YOLOAugmenter(data_dir=args.data_dir)
    
    # 1. Dọn dẹp nếu được yêu cầu
    if args.clean:
        augmenter.clean_augmented_files()
        
    # 2. Thực hiện sinh thêm ảnh tăng cường
    augmenter.generate_augmentations(target_count=args.target_count)
    
    # 3. Trực quan hóa mẫu so sánh
    if not args.no_viz:
        augmenter.save_visualizations(num_samples=10)

if __name__ == '__main__':
    main()
