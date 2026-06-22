import os
import cv2
import numpy as np
import joblib
from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import MiniBatchKMeans

# XGBoost cần OpenMP runtime (libomp) ở cấp hệ điều hành. Trên một số máy macOS
# chưa cài libomp, import sẽ lỗi. Để app vẫn chạy được RF/SVM, ta import tùy chọn.
try:
    from xgboost import XGBClassifier
    _XGBOOST_IMPORT_ERROR = None
except Exception as _e:  # pragma: no cover - phụ thuộc môi trường
    XGBClassifier = None
    _XGBOOST_IMPORT_ERROR = _e

class FeatureExtractorService(ABC):
    """
    Interface cho các dịch vụ trích xuất đặc trưng hình ảnh.
    """
    @abstractmethod
    def extract_features(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Trích xuất các đặc trưng (descriptors) từ ảnh.
        
        Args:
            image (np.ndarray): Ảnh đầu vào đọc bằng OpenCV (BGR hoặc Grayscale).
            mask (np.ndarray, optional): Mặt nạ xác định vùng quan tâm.
            
        Returns:
            np.ndarray: Ma trận descriptors có hình dạng (N, descriptor_dim) 
                        với N là số keypoint tìm thấy. Trả về mảng rỗng (0, descriptor_dim) nếu không tìm thấy keypoint.
        """
        pass

    @property
    @abstractmethod
    def descriptor_dim(self) -> int:
        """
        Số chiều của đặc trưng (ví dụ: 128 với SIFT, 32 với ORB).
        """
        pass


class SIFTService(FeatureExtractorService):
    """
    Dịch vụ trích xuất đặc trưng sử dụng thuật toán SIFT.
    """
    def __init__(self, n_features: int = 500):
        """
        Khởi tạo SIFT với số lượng đặc trưng tối đa.
        """
        self.sift = cv2.SIFT_create(nfeatures=n_features)

    def extract_features(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        # Chuyển đổi sang ảnh xám nếu là ảnh màu
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        keypoints, descriptors = self.sift.detectAndCompute(gray, mask)
        if descriptors is None:
            return np.zeros((0, self.descriptor_dim), dtype=np.float32)
        return descriptors.astype(np.float32)

    @property
    def descriptor_dim(self) -> int:
        return 128


class ORBService(FeatureExtractorService):
    """
    Dịch vụ trích xuất đặc trưng sử dụng thuật toán ORB (nhanh hơn SIFT).
    """
    def __init__(self, n_features: int = 500):
        """
        Khởi tạo ORB với số lượng đặc trưng tối đa.
        """
        self.orb = cv2.ORB_create(nfeatures=n_features)

    def extract_features(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        # Chuyển đổi sang ảnh xám nếu là ảnh màu
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        keypoints, descriptors = self.orb.detectAndCompute(gray, mask)
        if descriptors is None:
            return np.zeros((0, self.descriptor_dim), dtype=np.float32)
        return descriptors.astype(np.float32)

    @property
    def descriptor_dim(self) -> int:
        return 32


def extract_hsv_histogram(image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Trích xuất và chuẩn hóa L2 histogram màu sắc 3D HSV của ảnh.
    Đầu ra là vector phẳng 128 chiều (8 Hue x 4 Saturation x 4 Value).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Bins: 8 cho H, 4 cho S, 4 cho V
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, [8, 4, 4], [0, 180, 0, 256, 0, 256])
    hist_flat = hist.flatten()
    # Chuẩn hóa L2
    norm = np.linalg.norm(hist_flat, ord=2)
    if norm > 0:
        hist_flat /= norm
    return hist_flat

class VisualVocabulary:
    """
    Lớp biểu diễn BoVW (Bag-of-Visual-Words) từ điển hình ảnh dựa trên phân cụm K-Means.
    """
    def __init__(self, n_clusters: int = 300):
        self.n_clusters = n_clusters
        # Sử dụng MiniBatchKMeans để tối ưu hóa thời gian tính toán phân cụm
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000, n_init="auto")
        self.is_fitted = False

    def fit(self, descriptors_list: List[np.ndarray]) -> None:
        """
        Huấn luyện K-Means trên tất cả descriptors thu thập từ tập dữ liệu.
        """
        valid_descriptors = [d for d in descriptors_list if d is not None and len(d) > 0]
        if not valid_descriptors:
            raise ValueError("Không tìm thấy descriptors hợp lệ để huấn luyện Visual Vocabulary.")
        
        all_descriptors = np.vstack(valid_descriptors)
        self.kmeans.fit(all_descriptors)
        self.is_fitted = True

    def transform(self, descriptors: np.ndarray) -> np.ndarray:
        """
        Biểu diễn ma trận descriptors của một ảnh dưới dạng Histogram tần suất chuẩn hóa (BoVW vector).
        """
        histogram = np.zeros(self.n_clusters, dtype=np.float32)
        if not self.is_fitted:
            return histogram # Trả về vector zero nếu chưa fit (hoặc ném lỗi tùy logic)
        
        if descriptors is None or len(descriptors) == 0:
            return histogram
        
        # Tìm cụm từ vựng gần nhất cho mỗi descriptor
        words = self.kmeans.predict(descriptors)
        for w in words:
            histogram[w] += 1.0
            
        # Chuẩn hóa L1 (Histogram tần suất tương đối)
        norm = np.linalg.norm(histogram, ord=1)
        if norm > 0:
            histogram /= norm
        return histogram

    def save(self, filepath: str) -> None:
        """Lưu Visual Vocabulary vào file."""
        joblib.dump({
            "kmeans": self.kmeans,
            "n_clusters": self.n_clusters,
            "is_fitted": self.is_fitted
        }, filepath)

    def load(self, filepath: str) -> None:
        """Tải Visual Vocabulary từ file."""
        data = joblib.load(filepath)
        self.kmeans = data["kmeans"]
        self.n_clusters = data["n_clusters"]
        self.is_fitted = data["is_fitted"]


class ClassifierService(ABC):
    """
    Interface cho các dịch vụ phân loại ảnh hoa.
    """
    @abstractmethod
    def predict(self, feature_vector: np.ndarray) -> str:
        """
        Dự đoán lớp loài hoa từ vector đặc trưng.
        
        Args:
            feature_vector (np.ndarray): Vector đặc trưng của ảnh (BoVW histogram hoặc vector thô).
            
        Returns:
            str: Tên loài hoa dự đoán.
        """
        pass

    @abstractmethod
    def predict_proba(self, feature_vector: np.ndarray) -> np.ndarray:
        """
        Dự đoán xác suất phân loại lớp.
        """
        pass

    @abstractmethod
    def load(self, filepath: str) -> None:
        """Tải mô hình phân loại từ đĩa."""
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Lưu mô hình phân loại xuống đĩa."""
        pass


class RandomForestService(ClassifierService):
    """
    Dịch vụ phân loại sử dụng Random Forest.
    """
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = 12, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        self.classes: List[str] = []
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, classes: List[str]) -> None:
        """Huấn luyện mô hình Random Forest."""
        self.model.fit(X, y)
        self.classes = classes
        self.is_fitted = True

    def predict(self, feature_vector: np.ndarray) -> str:
        if not self.is_fitted:
            raise ValueError("Mô hình Random Forest chưa được huấn luyện (fit).")
        X = feature_vector.reshape(1, -1)
        pred_idx = self.model.predict(X)[0]
        return self.classes[int(pred_idx)]

    def predict_proba(self, feature_vector: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Mô hình Random Forest chưa được huấn luyện (fit).")
        X = feature_vector.reshape(1, -1)
        return self.model.predict_proba(X)[0]

    def load(self, filepath: str) -> None:
        data = joblib.load(filepath)
        self.model = data["model"]
        self.classes = data["classes"]
        self.is_fitted = data["is_fitted"]

    def save(self, filepath: str) -> None:
        joblib.dump({
            "model": self.model,
            "classes": self.classes,
            "is_fitted": self.is_fitted
        }, filepath)


class XGBoostService(ClassifierService):
    """
    Dịch vụ phân loại sử dụng XGBoost.
    """
    def __init__(self, max_depth: int = 4, learning_rate: float = 0.05, n_estimators: int = 100, random_state: int = 42):
        if XGBClassifier is None:
            raise RuntimeError(
                "Không dùng được XGBoost vì thiếu thư viện OpenMP (libomp) trên hệ thống. "
                "Cài bằng `brew install libomp` rồi thử lại, hoặc chọn RandomForest/SVM. "
                f"Chi tiết: {_XGBOOST_IMPORT_ERROR}"
            )
        self.model = XGBClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric='mlogloss'
        )
        self.classes: List[str] = []
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, classes: List[str]) -> None:
        """Huấn luyện mô hình XGBoost."""
        self.model.fit(X, y)
        self.classes = classes
        self.is_fitted = True

    def predict(self, feature_vector: np.ndarray) -> str:
        if not self.is_fitted:
            raise ValueError("Mô hình XGBoost chưa được huấn luyện (fit).")
        X = feature_vector.reshape(1, -1)
        pred_idx = self.model.predict(X)[0]
        return self.classes[int(pred_idx)]

    def predict_proba(self, feature_vector: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Mô hình XGBoost chưa được huấn luyện (fit).")
        X = feature_vector.reshape(1, -1)
        return self.model.predict_proba(X)[0]

    def load(self, filepath: str) -> None:
        data = joblib.load(filepath)
        self.model = data["model"]
        self.classes = data["classes"]
        self.is_fitted = data["is_fitted"]

    def save(self, filepath: str) -> None:
        joblib.dump({
            "model": self.model,
            "classes": self.classes,
            "is_fitted": self.is_fitted
        }, filepath)


class SVMService(ClassifierService):
    """
    Dịch vụ phân loại sử dụng SVM (Backward Compatibility với best_svm_flower.joblib).
    """
    def __init__(self):
        self.model = None
        self.classes: List[str] = []
        self.img_size: Tuple[int, int] = (32, 32)
        self.mean_image: np.ndarray = None
        self.scaler = None
        self.is_fitted = False

    def load(self, filepath: str) -> None:
        """
        Tải thông tin đóng gói SVM từ best_svm_flower.joblib.
        """
        data = joblib.load(filepath)
        self.model = data["model"]
        self.classes = data["classes"]
        self.img_size = data.get("img_size", (32, 32))
        self.mean_image = data["mean_image"]
        self.scaler = data["scaler"]
        self.is_fitted = True

    def save(self, filepath: str) -> None:
        if not self.is_fitted:
            raise ValueError("SVM model chưa được tải.")
        joblib.dump({
            "model": self.model,
            "classes": self.classes,
            "img_size": self.img_size,
            "mean_image": self.mean_image,
            "scaler": self.scaler,
            "model_type": "LinearSVC"
        }, filepath)

    def predict(self, feature_vector: np.ndarray) -> str:
        """
        Dự đoán lớp loài hoa từ vector ảnh thô đã làm phẳng dạng 1D (chưa chuẩn hóa).
        Hàm này tự động trừ ảnh trung bình (mean_image) và chuẩn hóa bằng scaler của SVM.
        """
        if not self.is_fitted:
            raise ValueError("SVM model chưa được tải.")
        
        # Chuyển đổi và định dạng vector thô thành (1, 3072)
        X = feature_vector.astype("float64").reshape(1, -1)
        # Thực hiện các bước chuẩn hóa tương ứng với pipeline gốc
        X = X - self.mean_image
        X = self.scaler.transform(X)
        
        pred_idx = self.model.predict(X)[0]
        return self.classes[int(pred_idx)]

    def predict_proba(self, feature_vector: np.ndarray) -> np.ndarray:
        """
        Vì LinearSVC không hỗ trợ predict_proba(), sử dụng decision_function và áp dụng Softmax
        để trả về phân phối xác suất dự kiến.
        """
        if not self.is_fitted:
            raise ValueError("SVM model chưa được tải.")
            
        X = feature_vector.astype("float64").reshape(1, -1)
        X = X - self.mean_image
        X = self.scaler.transform(X)
        
        scores = self.model.decision_function(X)[0]
        # Softmax
        exp_scores = np.exp(scores - np.max(scores))  # Ổn định số học tránh overflow
        probs = exp_scores / np.sum(exp_scores)
        return probs


class CNNService(ClassifierService):
    """
    Dịch vụ phân loại dùng CNN (Keras). Load model .h5, resize ảnh 64x64,
    normalize /255, predict trực tiếp từ ảnh BGR — không cần SIFT/ORB/vocab.
    """
    def __init__(self):
        self.model = None
        self.classes: List[str] = []
        self.is_fitted = False

    def load(self, filepath: str, classes: Optional[List[str]] = None) -> None:
        """Load model .h5 từ đĩa. classes có thể truyền vào hoặc lấy từ hub."""
        import tensorflow as tf
        self.model = tf.keras.models.load_model(filepath)
        if classes is not None:
            self.classes = classes
        self.is_fitted = True

    def save(self, filepath: str) -> None:
        if not self.is_fitted:
            raise ValueError("CNN model chưa được tải.")
        self.model.save(filepath)

    def _preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        """Resize về 64x64, chuyển BGR→RGB, normalize /255."""
        img = cv2.resize(image_bgr, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32) / 255.0

    def predict(self, image_bgr: np.ndarray) -> str:
        if not self.is_fitted:
            raise ValueError("CNN model chưa được tải.")
        x = self._preprocess(image_bgr)[np.newaxis, ...]  # (1, 64, 64, 3)
        probs = self.model.predict(x, verbose=0)[0]
        return self.classes[int(np.argmax(probs))]

    def predict_proba(self, image_bgr: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("CNN model chưa được tải.")
        x = self._preprocess(image_bgr)[np.newaxis, ...]
        return self.model.predict(x, verbose=0)[0]
