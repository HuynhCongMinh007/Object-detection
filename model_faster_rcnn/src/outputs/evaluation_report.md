# BÁO CÁO ĐÁNH GIÁ ĐỊNH LƯỢNG VÀ PHÂN TÍCH LỖI (ERROR ANALYSIS)

## 1. Chỉ số đánh giá chung cuộc trên tập Validation
- **Mean Precision (mPrecision):** 69.62%
- **Mean Recall (mRecall):** 85.87%
- **Mean F1-Score (mF1):** 76.59%
- **Điều kiện đối khớp (Matching Criteria):** IoU >= 0.50 & Score >= 0.50

## 2. Báo cáo Chi tiết từng Lớp Loài vật (Class-wise Performance)
| Loài vật (Class) | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Precision | Recall | F1-Score |
|---|---|---|---|---|---|---|
| cat | 61 | 34 | 12 | 64.21% | 83.56% | 72.62% |
| chicken | 99 | 32 | 13 | 75.57% | 88.39% | 81.48% |
| cow | 71 | 38 | 15 | 65.14% | 82.56% | 72.82% |
| dog | 75 | 53 | 6 | 58.59% | 92.59% | 71.77% |
| horse | 66 | 13 | 6 | 83.54% | 91.67% | 87.42% |
| sheep | 65 | 27 | 20 | 70.65% | 76.47% | 73.45% |

## 3. Phân tích Lỗi Nhầm lẫn Hàng đầu (Top 3 Confusions)
Dưới đây là 3 cặp loài vật mà mô hình dễ nhận diện nhầm lẫn cho nhau nhất:
1. **Thực tế:** `sheep` nhưng mô hình nhận nhầm thành `dog`: **6 lần**
2. **Thực tế:** `cat` nhưng mô hình nhận nhầm thành `dog`: **5 lần**
3. **Thực tế:** `dog` nhưng mô hình nhận nhầm thành `cat`: **3 lần**

## 4. Phân tích Lỗi Bỏ sót (False Negatives) & Nhận nhầm Nền (False Positives)
- **Lỗi Bỏ Sót vật thể (Missed Objects):**
  + Lớp `cat` bị bỏ sót hoàn toàn (dự đoán là Nền): **7 lần**
  + Lớp `chicken` bị bỏ sót hoàn toàn (dự đoán là Nền): **9 lần**
  + Lớp `cow` bị bỏ sót hoàn toàn (dự đoán là Nền): **9 lần**
  + Lớp `dog` bị bỏ sót hoàn toàn (dự đoán là Nền): **1 lần**
  + Lớp `horse` bị bỏ sót hoàn toàn (dự đoán là Nền): **3 lần**
  + Lớp `sheep` bị bỏ sót hoàn toàn (dự đoán là Nền): **10 lần**

- **Lỗi Nhận diện sai Nền (Hallucinations/False Alarms):**
  + Vùng Nền trống bị mô hình dự đoán nhầm thành `cat`: **30 lần**
  + Vùng Nền trống bị mô hình dự đoán nhầm thành `chicken`: **32 lần**
  + Vùng Nền trống bị mô hình dự đoán nhầm thành `cow`: **35 lần**
  + Vùng Nền trống bị mô hình dự đoán nhầm thành `dog`: **38 lần**
  + Vùng Nền trống bị mô hình dự đoán nhầm thành `horse`: **8 lần**
  + Vùng Nền trống bị mô hình dự đoán nhầm thành `sheep`: **21 lần**

## 5. Heatmap Ma trận nhầm lẫn trực quan
![Confusion Matrix Heatmap](confusion_matrix.png)
