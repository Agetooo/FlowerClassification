import os
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import cv2
import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    _XGBOOST_IMPORT_ERROR = None
except Exception as _e:  # pragma: no cover - depends on local runtime
    XGBClassifier = None
    _XGBOOST_IMPORT_ERROR = _e


class FeatureExtractorService(ABC):
    @abstractmethod
    def extract_features(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def descriptor_dim(self) -> int:
        pass


class SIFTService(FeatureExtractorService):
    def __init__(self, n_features: int = 500):
        self.sift = cv2.SIFT_create(nfeatures=n_features)

    def extract_features(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        _, descriptors = self.sift.detectAndCompute(gray, mask)
        if descriptors is None:
            return np.zeros((0, self.descriptor_dim), dtype=np.float32)
        return descriptors.astype(np.float32)

    @property
    def descriptor_dim(self) -> int:
        return 128


class ORBService(FeatureExtractorService):
    def __init__(self, n_features: int = 500):
        self.orb = cv2.ORB_create(nfeatures=n_features)

    def extract_features(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        _, descriptors = self.orb.detectAndCompute(gray, mask)
        if descriptors is None:
            return np.zeros((0, self.descriptor_dim), dtype=np.float32)
        return descriptors.astype(np.float32)

    @property
    def descriptor_dim(self) -> int:
        return 32


class HOGService(FeatureExtractorService):
    """
    Feature extractor using local HOG block descriptors.
    """

    def __init__(
        self,
        win_size: Tuple[int, int] = (128, 128),
        cell_size: Tuple[int, int] = (8, 8),
        block_size: Tuple[int, int] = (16, 16),
        block_stride: Tuple[int, int] = (8, 8),
        nbins: int = 9,
    ):
        self.win_size = win_size
        self.block_size = block_size
        self.block_stride = block_stride
        self.cell_size = cell_size
        self.nbins = nbins
        self.hog = cv2.HOGDescriptor(
            _winSize=win_size,
            _blockSize=block_size,
            _blockStride=block_stride,
            _cellSize=cell_size,
            _nbins=nbins,
        )

    def extract_features(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        gray = cv2.resize(gray, self.win_size, interpolation=cv2.INTER_LINEAR)

        if mask is not None:
            resized_mask = cv2.resize(mask, self.win_size, interpolation=cv2.INTER_NEAREST)
            gray = cv2.bitwise_and(gray, gray, mask=resized_mask)

        descriptors = self.hog.compute(gray)
        if descriptors is None:
            return np.zeros((0, self.descriptor_dim), dtype=np.float32)

        return descriptors.reshape(-1, self.descriptor_dim).astype(np.float32)

    @property
    def descriptor_dim(self) -> int:
        cells_per_block_x = self.block_size[0] // self.cell_size[0]
        cells_per_block_y = self.block_size[1] // self.cell_size[1]
        return cells_per_block_x * cells_per_block_y * self.nbins


def extract_hsv_histogram(image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, [8, 4, 4], [0, 180, 0, 256, 0, 256])
    hist_flat = hist.flatten()
    norm = np.linalg.norm(hist_flat, ord=2)
    if norm > 0:
        hist_flat /= norm
    return hist_flat


class VisualVocabulary:
    def __init__(self, n_clusters: int = 300):
        self.n_clusters = n_clusters
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000, n_init="auto")
        self.is_fitted = False

    def fit(self, descriptors_list: List[np.ndarray]) -> None:
        valid_descriptors = [d for d in descriptors_list if d is not None and len(d) > 0]
        if not valid_descriptors:
            raise ValueError("No valid descriptors found to train the visual vocabulary.")

        self.kmeans.fit(np.vstack(valid_descriptors))
        self.is_fitted = True

    def transform(self, descriptors: np.ndarray) -> np.ndarray:
        histogram = np.zeros(self.n_clusters, dtype=np.float32)
        if not self.is_fitted or descriptors is None or len(descriptors) == 0:
            return histogram

        words = self.kmeans.predict(descriptors)
        for word in words:
            histogram[word] += 1.0

        norm = np.linalg.norm(histogram, ord=1)
        if norm > 0:
            histogram /= norm
        return histogram

    def save(self, filepath: str) -> None:
        joblib.dump(
            {
                "kmeans": self.kmeans,
                "n_clusters": self.n_clusters,
                "is_fitted": self.is_fitted,
            },
            filepath,
        )

    def load(self, filepath: str) -> None:
        data = joblib.load(filepath)
        self.kmeans = data["kmeans"]
        self.n_clusters = data["n_clusters"]
        self.is_fitted = data["is_fitted"]


class ClassifierService(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, classes: List[str]) -> None:
        pass

    @abstractmethod
    def predict(self, feature_vector: np.ndarray) -> str:
        pass

    @abstractmethod
    def predict_proba(self, feature_vector: np.ndarray) -> np.ndarray:
        pass

    def predict_with_confidence(self, feature_vector: np.ndarray) -> dict:
        probs = self.predict_proba(feature_vector)
        pred_idx = int(np.argmax(probs))
        return {
            "predicted_class": self.classes[pred_idx],
            "confidence": float(probs[pred_idx]),
        }

    @abstractmethod
    def load(self, filepath: str) -> None:
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        pass


class SklearnClassifierService(ClassifierService):
    def __init__(self):
        self.model = None
        self.classes: List[str] = []
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, classes: List[str]) -> None:
        self.model.fit(X, y)
        self.classes = classes
        self.is_fitted = True

    def predict(self, feature_vector: np.ndarray) -> str:
        if not self.is_fitted:
            raise ValueError(f"{self.__class__.__name__} has not been fitted.")
        pred_idx = self.model.predict(feature_vector.reshape(1, -1))[0]
        return self.classes[int(pred_idx)]

    def predict_proba(self, feature_vector: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError(f"{self.__class__.__name__} has not been fitted.")
        return self.model.predict_proba(feature_vector.reshape(1, -1))[0]

    def load(self, filepath: str) -> None:
        data = joblib.load(filepath)
        self.model = data["model"]
        self.classes = data["classes"]
        self.is_fitted = data["is_fitted"]

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "classes": self.classes,
                "is_fitted": self.is_fitted,
                "model_type": self.__class__.__name__,
            },
            filepath,
        )


class RandomForestService(SklearnClassifierService):
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = 12, random_state: int = 42):
        super().__init__()
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=4,
            max_features="sqrt",
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )


class XGBoostService(SklearnClassifierService):
    def __init__(
        self,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
    ):
        super().__init__()
        if XGBClassifier is None:
            raise RuntimeError(
                "Cannot use XGBoost because the OpenMP runtime is missing. "
                f"Details: {_XGBOOST_IMPORT_ERROR}"
            )
        self.model = XGBClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric="mlogloss",
        )


class SoftmaxService(SklearnClassifierService):
    def __init__(self, random_state: int = 42):
        super().__init__()
        self.model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SGDClassifier(
                        loss="log_loss",
                        penalty="l2",
                        alpha=0.01,
                        max_iter=2000,
                        tol=1e-4,
                        random_state=random_state,
                    ),
                ),
            ]
        )


class SVMService(SklearnClassifierService):
    def __init__(self):
        super().__init__()
        self.model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        C=1,
                        gamma="scale",
                        probability=True,
                    ),
                ),
            ]
        )


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

    def fit(self, X: np.ndarray, y: np.ndarray, classes: List[str]) -> None:
        raise NotImplementedError("CNN training dùng train_cnn.py riêng, không train qua hub.")

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
        x = self._preprocess(image_bgr)[np.newaxis, ...]
        probs = self.model.predict(x, verbose=0)[0]
        return self.classes[int(np.argmax(probs))]

    def predict_proba(self, image_bgr: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("CNN model chưa được tải.")
        x = self._preprocess(image_bgr)[np.newaxis, ...]
        return self.model.predict(x, verbose=0)[0]


class KNNService(SklearnClassifierService):
    def __init__(self, n_neighbors: int = 5):
        super().__init__()
        self.model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=n_neighbors)),
            ]
        )
