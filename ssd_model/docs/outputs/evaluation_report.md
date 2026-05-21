# KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH SSD

- **Tổng số ảnh test:** 172
- **IoU Threshold:** 0.5
- **Confidence Threshold:** 0.5
- **Average FPS:** **39.06 frames/sec**

### Hiệu năng chi tiết (Class-wise Metrics)

| Lớp động vật | TP | FP | FN | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **cat** | 29 | 12 | 11 | 70.73% | 72.50% | 71.60% |
| **cat-chicken-cow-dog-horse-sheep** | 0 | 0 | 0 | 0.00% | 0.00% | 0.00% |
| **chicken** | 38 | 10 | 8 | 79.17% | 82.61% | 80.85% |
| **cow** | 43 | 13 | 20 | 76.79% | 68.25% | 72.27% |
| **dog** | 35 | 10 | 8 | 77.78% | 81.40% | 79.55% |
| **horse** | 26 | 4 | 6 | 86.67% | 81.25% | 83.87% |
| **sheep** | 26 | 4 | 18 | 86.67% | 59.09% | 70.27% |

### Chỉ số Trung bình (Mean Metrics)
- **Mean Precision (mP):** **68.26%**
- **Mean Recall (mR):** **63.59%**
- **Mean F1-Score:** **65.49%**