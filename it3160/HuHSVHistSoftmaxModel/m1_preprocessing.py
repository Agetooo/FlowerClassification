import cv2
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


def extract_features(image_path):
    """
    Trích xuất đặc trưng Hu Moments + HSV Histogram từ 1 ảnh
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Không thể đọc ảnh hoặc file lỗi: {image_path}")

    # --- 1. HSV Histogram ---
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [8, 8, 4], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    hist_features = hist.flatten()

    # --- 2. Hu Moments ---
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    moments = cv2.moments(thresh)
    hu_moments = cv2.HuMoments(moments).flatten()
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

    return np.hstack([hist_features, hu_moments])


def process_image_folders(dataset_root_path, output_features_npy, output_labels_npy, output_classes_npy):
    """
    Duyệt folder, trích xuất đặc trưng, mã hóa nhãn và lưu file.
    """
    dataset_path = Path(dataset_root_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Thư mục {dataset_root_path} không tồn tại!")

    X_list = []
    y_string_list = []

    print(f"Bắt đầu quét thư mục: {dataset_root_path} ...")

    # 1. Duyệt qua tất cả các file trong tất cả thư mục con
    # (Hỗ trợ nhiều định dạng ảnh phổ biến)
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    for img_path in dataset_path.rglob('*'):
        if img_path.is_file() and img_path.suffix.lower() in valid_extensions:
            # Lấy tên của folder cha chứa file này làm nhãn
            class_name = img_path.parent.name

            try:
                # Trích xuất vector
                features = extract_features(img_path)
                X_list.append(features)
                y_string_list.append(class_name)
            except Exception as e:
                print(f"Bỏ qua {img_path.name}: {e}")

    # 2. Chuyển đổi list thành numpy array
    X = np.array(X_list, dtype=np.float32)
    y_strings = np.array(y_string_list)

    print(f"\nĐã trích xuất xong đặc trưng của {len(X)} ảnh.")

    # 3. Mã hóa nhãn (String -> Integer 0, 1, 2...)
    print("Đang mã hóa nhãn lớp...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_strings)

    # Lấy danh sách tên các lớp (Để sau này map ngược từ số ra tên)
    class_names = label_encoder.classes_

    for idx, name in enumerate(class_names):
        print(f" - Lớp '{name}' được mã hóa thành số: {idx}")

    # 4. Lưu ra file
    np.save(output_features_npy, X)
    np.save(output_labels_npy, y_encoded)
    np.save(output_classes_npy, class_names)  # Cực kỳ quan trọng để inference sau này

    print(f"\n[Hoàn tất] Đã lưu:")
    print(f" -> Dữ liệu X: {output_features_npy} (Kích thước: {X.shape})")
    print(f" -> Nhãn y: {output_labels_npy} (Kích thước: {y_encoded.shape})")
    print(f" -> Từ điển nhãn: {output_classes_npy}")

    return X, y_encoded, class_names


# ==========================================
# CÁCH SỬ DỤNG
# ==========================================
# Cấu hình đường dẫn
ROOT_FOLDER = "flowers"  # tên thư mục chứa các folder con của bạn
X_FILE = "X_features.npy"
Y_FILE = "y_labels.npy"
CLASSES_FILE = "classes_mapping.npy"

# Gọi hàm chạy (Chỉ cần chạy 1 lần)
X, y, class_names = process_image_folders(ROOT_FOLDER, X_FILE, Y_FILE, CLASSES_FILE)