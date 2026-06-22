# FlowerClassification

Ứng dụng web phân loại hoa sử dụng các kỹ thuật Machine Learning cổ điển, được xây dựng bằng Flask. Hỗ trợ nhiều pipeline trích xuất đặc trưng và phân loại khác nhau có thể chuyển đổi linh hoạt ngay trên giao diện.

---

## Tính năng

- Phân loại **7 loài hoa**: Bellflower, Daisy, Dandelion, Lotus, Rose, Sunflower, Tulip
- **4 bộ phân loại** có thể chọn: Random Forest, XGBoost, SVM, Softmax
- **3 thuật toán trích xuất đặc trưng**: SIFT, ORB và HOG (dùng với RF và XGBoost)
- Hiển thị **phân phối xác suất** dự đoán của từng loài hoa
- So sánh ảnh gốc và ảnh đã tiền xử lý trên UI
- Hỗ trợ **huấn luyện mô hình mới** trực tiếp từ giao diện web (chạy nền)

---

## Kiến trúc hệ thống

```
FlowerClassification/
├── app.py                      # Flask server – API endpoints
├── hub.py                      # ClassificationHub (Strategy + Factory Pattern)
├── services.py                 # Các service trích xuất đặc trưng & phân loại
├── preprocess.py               # Tiền xử lý ảnh (resize, blur, flower mask)
├── templates/
│   └── index.html              # Giao diện web
├── it3160/
│   ├── HuHSVHistSoftmaxModel/  # Model Softmax pre-trained (Hu + HSV)
│   ├── best_softmax.npy
│   ├── random_forest_flower.py
│   └── softmax.ipynb
├── best_svm_flower.joblib      # Model SVM pre-trained (raw pixels 32x32)
├── vocab_sift_opt.joblib       # Visual Vocabulary SIFT (K=300)
├── vocab_orb_opt.joblib        # Visual Vocabulary ORB (K=300)
├── vocab_hog_opt.joblib        # Visual Vocabulary HOG (K=300)
├── model_randomforest_sift_opt.joblib
├── model_randomforest_orb_opt.joblib
├── model_xgboost_sift_opt.joblib
├── model_xgboost_orb_opt.joblib
├── flower-training/            # Dataset (symlink hoặc thư mục thực)
│   ├── bellflower/
│   ├── daisy/
│   ├── dandelion/
│   ├── lotus/
│   ├── rose/
│   ├── sunflower/
│   └── tulip/
└── requirements.txt
```

### Luồng xử lý

```
Ảnh đầu vào
    │
    ▼
Tiền xử lý (resize 256×256, Gaussian blur, flower mask)
    │
    ├─── SVM ──────► Raw pixels (32×32 = 3072 chiều) ──► LinearSVC
    │
    ├─── Softmax ──► Hu Moments (7) + HSV Histogram (256) = 263 chiều ──► W·x + b
    │
    └─── SIFT/ORB/HOG ─► Descriptors ──► BoVW K-Means (K=300)
                                          + HSV Histogram (128)
                                          = Fused Vector (428 chiều)
                                               │
                                    ┌──────────┴──────────┐
                                    ▼                     ▼
                              Random Forest           XGBoost
```

---

## Cài đặt

### Yêu cầu hệ thống

- Python 3.9+
- macOS / Linux (đã kiểm thử trên macOS Apple Silicon)
- `libomp` (cần thiết cho XGBoost trên macOS): `brew install libomp`

### Bước cài đặt

**1. Clone repository và tạo môi trường ảo:**

```bash
git clone <repo-url>
cd FlowerClassification
python3 -m venv venv
source venv/bin/activate
```

**2. Cài đặt các thư viện:**

```bash
pip install -r requirements.txt
```

**3. Chuẩn bị dataset:**

Dataset cần được đặt (hoặc tạo symlink) tại `flower-training/` với cấu trúc:

```
flower-training/
├── bellflower/   # ~300–500 ảnh .jpg/.png
├── daisy/
├── dandelion/
├── lotus/
├── rose/
├── sunflower/
└── tulip/
```

Ví dụ tạo symlink nếu dataset ở nơi khác:

```bash
ln -s /đường/dẫn/tới/dataset flower-training
```

**4. Chạy ứng dụng:**

```bash
python app.py
```

Mở trình duyệt tại: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## Sử dụng

### Dự đoán ảnh

1. Mở giao diện web
2. Chọn **Feature Extractor** (SIFT / ORB / HOG) — chỉ dùng cho RF và XGBoost
3. Chọn **Classifier** (Random Forest / XGBoost / SVM / Softmax)
4. Upload ảnh hoa
5. Nhấn **Predict** — kết quả hiển thị kèm biểu đồ xác suất và so sánh ảnh tiền xử lý

> **Lưu ý:** SVM và Softmax là các model pre-trained, không cần chọn Feature Extractor.

### Huấn luyện mô hình mới

1. Chọn cấu hình Extractor + Classifier (RF hoặc XGBoost)
2. Nhấn **Train** — quá trình chạy nền, có thể theo dõi tiến trình real-time
3. Sau khi hoàn tất, model được lưu tự động và sẵn sàng dùng để predict

---

## Các Pipeline phân loại

| Classifier   | Feature Extractor | Đặc trưng                         | Ghi chú                        |
|--------------|-------------------|------------------------------------|---------------------------------|
| Random Forest | SIFT, ORB hoặc HOG | BoVW (300) + HSV Histogram (128)   | Cần train hoặc load model       |
| XGBoost      | SIFT, ORB hoặc HOG | BoVW (300) + HSV Histogram (128)   | Yêu cầu `libomp` trên macOS     |
| SVM          | Không dùng        | Raw pixels 32×32 = 3072 chiều      | Pre-trained, không train lại    |
| Softmax      | Không dùng        | Hu Moments (7) + HSV (256) = 263 chiều | Pre-trained, không train lại |

---

## API Endpoints

| Method | Endpoint        | Mô tả                                      |
|--------|-----------------|--------------------------------------------|
| GET    | `/`             | Giao diện web chính                        |
| POST   | `/predict`      | Dự đoán loài hoa từ ảnh upload             |
| POST   | `/train`        | Bắt đầu huấn luyện mô hình (chạy nền)      |
| GET    | `/train_status` | Theo dõi trạng thái huấn luyện             |

### `/predict` – Request

```
Content-Type: multipart/form-data
image:      <file ảnh>
extractor:  "SIFT" | "ORB" | "HOG"   (mặc định: "ORB")
classifier: "RandomForest" | "XGBoost" | "SVM" | "Softmax"  (mặc định: "RandomForest")
```

### `/predict` – Response

```json
{
  "status": "success",
  "predicted_class": "rose",
  "probabilities": {
    "bellflower": 0.02,
    "daisy": 0.05,
    "dandelion": 0.01,
    "lotus": 0.03,
    "rose": 0.82,
    "sunflower": 0.04,
    "tulip": 0.03
  },
  "original_image": "data:image/jpeg;base64,...",
  "preprocessed_image": "data:image/jpeg;base64,...",
  "config": {
    "extractor": "ORB",
    "classifier": "RandomForest"
  }
}
```

---

## Xử lý sự cố

**XGBoost báo lỗi khi import:**

```bash
brew install libomp
# Hoặc chạy script có sẵn:
bash fix_xgboost_openmp.sh
```

**Lỗi "Mô hình chưa được huấn luyện":**

- Kiểm tra các file `.joblib` đã có đủ chưa (xem mục cấu trúc thư mục)
- Hoặc sử dụng tính năng Train trên giao diện web để tạo mô hình mới

**Dataset không tìm thấy khi train:**

- Đảm bảo thư mục `flower-training/` tồn tại và có đủ 7 thư mục con tương ứng với 7 loài hoa

---

## Công nghệ sử dụng

- **Web framework**: Flask
- **Xử lý ảnh**: OpenCV, Pillow
- **Machine Learning**: scikit-learn (Random Forest, K-Means, SVM), XGBoost
- **Số học / Mảng**: NumPy
- **Lưu/tải model**: joblib

---

## Môn học

Đồ án môn **IT3160 – Nhập môn Học Máy**, Trường Đại học Bách Khoa Hà Nội.
