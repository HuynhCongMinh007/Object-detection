import os
import sys
import tempfile
import streamlit as st
import numpy as np
from PIL import Image
import torch
import cv2

# Thêm đường dẫn tới thư mục gốc (chứa Yolov8Nano_model) vào hệ thống
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO_MODEL_DIR = os.path.join(BASE_DIR, 'Yolov8Nano_model')
sys.path.append(YOLO_MODEL_DIR)

# Import các hàm từ Yolov8Nano_model/inference.py
from inference import (
    load_model, 
    run_inference
)

# Đường dẫn tuyệt đối đến tệp mô hình
MODEL_PATH = os.path.join(YOLO_MODEL_DIR, 'checkpoints', 'best.pt')

# Thiết lập cho ứng dụng
st.set_page_config(page_title="AI Vision - YOLOv8", page_icon="✨", layout="wide")

st.title("✨ Khám Phá AI Vision")
st.markdown("##### Trải nghiệm sức mạnh nhận diện đối tượng của YOLOv8 Nano ngay trên trình duyệt của bạn")

# ==========================================
# Cấu hình thanh Sidebar
# ==========================================
with st.sidebar:
    # Thay thế logo bằng component HTML nội bộ để không bị lỗi link ảnh
    st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h1 style="margin-bottom: 5px; font-weight: 900; letter-spacing: 2px; background: -webkit-linear-gradient(45deg, #FF4B4B, #FF8A00); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">DETECTION</h1>
            <p style="color: gray; font-style: italic; margin-top: 5px;">Phiên bản Nano siêu nhẹ</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### ⚙️ Bảng Điều Khiển")
    st.markdown("Tinh chỉnh các tham số AI theo ý muốn:")
    
    conf_thresh = st.slider(
        "🎚️ Ngưỡng tin cậy (Confidence)", 
        min_value=0.1, max_value=1.0, value=0.5, step=0.05,
        help="Lọc các dự đoán có độ chắc chắn thấp nhằm tránh nhận diện nhầm."
    )
    
    st.markdown("---")
    st.info("💡 **Mẹo:** Tải lên ảnh có độ phân giải tốt, sáng rõ để AI dự đoán chính xác nhất!")
    st.success("🤖 **Mô hình Core:** `YOLOv8 Nano`")

# ==========================================
# Khởi tạo mô hình
# ==========================================
@st.cache_resource
def load_detection_model():
    # Cache lại để không phải nạp mô hình lại mỗi lần reload UI
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Lỗi khi nạp mô hình: {e}")
        return None

model = load_detection_model()

if model is None:
    st.error("🚨 Vui lòng kiểm tra lại đường dẫn model. Chưa thể khởi chạy!")
    st.stop()


# ==========================================
# Giao diện tải ảnh & Hiển thị
# ==========================================
st.markdown("#### 📤 Vui lòng tải một bức ảnh lên đây:")
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], help="Kéo thả ảnh hoặc nhấp để tải từ máy của bạn.")

if uploaded_file is None:
    st.info("👆 Đang chờ... Bạn hãy tải một bức ảnh lên để AI có thể trổ tài nhé.")
else:
    # 1. Nạp ảnh
    image = Image.open(uploaded_file).convert("RGB")
    
    # Tạo layout 2 cột rõ ràng, có gap giúp thoáng đãng
    col_img, col_res = st.columns([1, 1], gap="large")
    
    with col_img:
        st.markdown("##### 🖼️ Ảnh Gốc")
        st.image(image, use_container_width=True)
        
        # Đặt nút bấm nổi bật dưới ảnh gốc
        run_btn = st.button("🚀 BẮT ĐẦU PHÂN TÍCH NGAY", type="primary", use_container_width=True)
        
    # 2. Xử lý & Phân tích
    with col_res:
        st.markdown("##### 🎯 Kết Quả Nhận Diện")
        
        if run_btn:
            with st.spinner("⏳ Khởi động Neural Network... Đang soi từng pixel..."):
                try:
                    # Chạy logic nhận diện YOLOv8
                    result = run_inference(
                        model=model, 
                        image_path=image, 
                        conf_threshold=conf_thresh
                    )
                    
                    num_detections = result.get('num_detections', 0)
                    
                    if num_detections == 0:
                        st.warning("😕 Rất tiếc, AI không tìm thấy đối tượng quen thuộc nào trong khung hình (hoặc có thể thử hạ Ngưỡng tin cậy xuống).")
                        st.image(image, use_container_width=True)
                    else:
                        st.balloons() # Bắn bóng bay ăn mừng
                        
                        # Vẽ bounding box bằng YOLO engine
                        results_yolo = model.predict(source=image, conf=conf_thresh, save=False)
                        annotated_img_bgr = results_yolo[0].plot(line_width=2, font_size=4)
                        annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)
                        
                        # CẦN ĐƯA ẢNH LÊN TRƯỚC ĐỂ ĐỒNG BẰNG VỚI ẢNH GỐC BÊN TRÁI
                        st.image(annotated_img_rgb, use_container_width=True, caption="Kết quả Bounding Boxes (Khung giới hạn)")
                        
                        # Hiển thị số lượng trực quan phía DƯỚI ảnh (ngang hàng với nút phân tích bên trái)
                        st.success(f"🎉 Phát hiện thành công **{num_detections}** vật thể!")
                        
                        # Ẩn bớt data thô JSON cho giao diện thân thiện với Non-Dev
                        with st.expander("🛠 Xem Dữ Liệu Phân Tích Thô (Dành cho Dev / Nghiên cứu)"):
                            st.json(result)
                            
                except Exception as e:
                    st.error(f"❌ Xảy ra lỗi bất ngờ: {e}")
        else:
            st.info("👈 Bấm chọn **BẮT ĐẦU PHÂN TÍCH NGAY** để kích hoạt trí tuệ nhân tạo nha!")
