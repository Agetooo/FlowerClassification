import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from preprocess import extract_flower_mask, preprocess_single_image
from services import (
    ClassifierService,
    CNNService,
    FeatureExtractorService,
    HOGService,
    KNNService,
    ORBService,
    RandomForestService,
    SIFTService,
    SVMService,
    VisualVocabulary,
    XGBoostService,
    extract_hsv_histogram,
)


class ClassificationHub:
    """
    Coordinates feature extraction, BoVW vectorization, classifier training,
    and inference for flower classification.
    """

    def __init__(self, models_dir: str = "."):
        self.models_dir = models_dir
        self.extractor: Optional[FeatureExtractorService] = None
        self.classifier: Optional[ClassifierService] = None
        self.vocab: Optional[VisualVocabulary] = None
        self.extractor_name: str = ""
        self.classifier_name: str = ""
        self.classes = ["bellflower", "daisy", "dandelion", "lotus", "rose", "sunflower", "tulip"]

    def set_extractor(self, extractor_type: str) -> None:
        name = extractor_type.upper()
        if name == "SIFT":
            self.extractor = SIFTService()
            self.extractor_name = "SIFT"
        elif name == "ORB":
            self.extractor = ORBService()
            self.extractor_name = "ORB"
        elif name == "HOG":
            self.extractor = HOGService()
            self.extractor_name = "HOG"
        else:
            raise ValueError(f"Unsupported feature extractor: {extractor_type}")

        self.vocab = VisualVocabulary(n_clusters=500)
        self._auto_load_if_possible()

    def set_classifier(self, classifier_type: str) -> None:
        """
        Khởi tạo Classifier Service dựa trên Factory Pattern.

        Args:
            classifier_type (str): 'RandomForest', 'XGBoost', 'SVM', 'KNN' hoặc 'CNN'
        """
        name = classifier_type.upper()
        if name == "RANDOMFOREST":
            self.classifier = RandomForestService()
            self.classifier_name = "RandomForest"
        elif name == "XGBOOST":
            self.classifier = XGBoostService()
            self.classifier_name = "XGBoost"
        elif name == "SVM":
            self.classifier = SVMService()
            self.classifier_name = "SVM"
        elif name == "KNN":
            self.classifier = KNNService()
            self.classifier_name = "KNN"
        elif name == "CNN":
            self.classifier = CNNService()
            self.classifier_name = "CNN"
        else:
            raise ValueError(f"Không hỗ trợ thuật toán phân loại: {classifier_type}")

        self._auto_load_if_possible()

    def _get_vocab_path(self) -> str:
        return os.path.join(self.models_dir, f"vocab_{self.extractor_name.lower()}_opt.joblib")

    def _get_classifier_path(self) -> str:
        if self.classifier_name == "CNN":
            return os.path.join(self.models_dir, "cnn_flower_model.h5")
        return os.path.join(
            self.models_dir,
            f"model_{self.classifier_name.lower()}_{self.extractor_name.lower()}_opt.joblib",
        )

    def _auto_load_if_possible(self) -> None:
        if self.classifier_name == "CNN":
            path = self._get_classifier_path()
            if os.path.exists(path):
                self.classifier.load(path, classes=self.classes)
                print(f"[Hub] Loaded CNN model from: {path}")
            return

        if not (self.extractor_name and self.classifier_name and self.vocab and self.classifier):
            return

        vocab_path = self._get_vocab_path()
        clf_path = self._get_classifier_path()
        if os.path.exists(vocab_path) and os.path.exists(clf_path):
            self.vocab.load(vocab_path)
            self.classifier.load(clf_path)
            print(f"[Hub] Loaded pipeline: {self.extractor_name} + {self.classifier_name}")

    def _resolve_dataset_dir(self, dataset_dir: str) -> str:
        if os.path.exists(dataset_dir):
            return dataset_dir

        fallback = os.path.join("it3160", dataset_dir)
        if os.path.exists(fallback):
            return fallback

        raise FileNotFoundError(f"Training dataset not found: {dataset_dir}")

    def _load_image_paths(self, dataset_dir: str) -> Tuple[List[str], List[int]]:
        dataset_root = self._resolve_dataset_dir(dataset_dir)
        image_paths: List[str] = []
        labels: List[int] = []

        for label_idx, cls in enumerate(self.classes):
            cls_dir = os.path.join(dataset_root, cls)
            if not os.path.exists(cls_dir):
                print(f"Skipping missing class folder: {cls_dir}")
                continue

            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    image_paths.append(os.path.join(cls_dir, img_name))
                    labels.append(label_idx)

        if not image_paths:
            raise ValueError(f"No training images found in: {dataset_root}")

        return image_paths, labels

    def _extract_descriptors_and_color(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        cleaned_img = preprocess_single_image(img, targetedSize=(256, 256))
        mask = extract_flower_mask(cleaned_img)
        descriptors = self.extractor.extract_features(cleaned_img, mask=mask)
        color_hist = extract_hsv_histogram(cleaned_img, mask=mask)
        return descriptors, color_hist

    def _fuse_features(self, descriptors: np.ndarray, color_hist: np.ndarray) -> np.ndarray:
        hist = self.vocab.transform(descriptors)
        hist_l2 = hist / (np.linalg.norm(hist, ord=2) + 1e-6)
        color_l2 = color_hist / (np.linalg.norm(color_hist, ord=2) + 1e-6)
        feature_vector = np.hstack([hist_l2, 0.5 * color_l2])
        feature_vector /= np.linalg.norm(feature_vector, ord=2) + 1e-6
        return feature_vector

    def train_pipeline(self, dataset_dir: str) -> None:
        if self.classifier_name == "CNN":
            print("[Hub] CNN sử dụng train_cnn.py riêng, không train từ UI.")
            return

        if not self.extractor_name or not self.classifier_name or not self.extractor or not self.classifier:
            raise ValueError("Set both extractor and classifier before training.")

        print(f"\n--- Training pipeline: {self.extractor_name} + {self.classifier_name} ---")
        image_paths, labels = self._load_image_paths(dataset_dir)
        print(f"Collected {len(image_paths)} training images.")

        descriptors_list: List[np.ndarray] = []
        color_hists_list: List[np.ndarray] = []
        valid_labels: List[int] = []

        for idx, img_path in enumerate(image_paths):
            try:
                descriptors, color_hist = self._extract_descriptors_and_color(img_path)
            except ValueError as exc:
                print(f"Skipping {img_path}: {exc}")
                continue

            descriptors_list.append(descriptors)
            color_hists_list.append(color_hist)
            valid_labels.append(labels[idx])

            if (idx + 1) % 500 == 0 or (idx + 1) == len(image_paths):
                print(f"Processed {idx + 1}/{len(image_paths)} images.")

        y = np.array(valid_labels)
        indices = np.arange(len(y))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=y, random_state=42)
        print(f"[Hub] Split: {len(train_idx)} train, {len(test_idx)} test.")

        train_descriptors = [descriptors_list[i] for i in train_idx]
        self.vocab.fit(train_descriptors)
        vocab_path = self._get_vocab_path()
        self.vocab.save(vocab_path)
        print(f"Saved visual vocabulary: {vocab_path}")

        X_train = np.array([self._fuse_features(descriptors_list[i], color_hists_list[i]) for i in train_idx])
        y_train = y[train_idx]
        X_test = np.array([self._fuse_features(descriptors_list[i], color_hists_list[i]) for i in test_idx])
        y_test = y[test_idx]

        print(f"Training classifier: {self.classifier_name}")
        self.classifier.fit(X_train, y_train, classes=self.classes)
        clf_path = self._get_classifier_path()
        self.classifier.save(clf_path)
        print(f"Saved classifier: {clf_path}")

        train_preds = np.array([self.classes.index(self.classifier.predict(x)) for x in X_train])
        test_preds = np.array([self.classes.index(self.classifier.predict(x)) for x in X_test])
        print("\n==================================================")
        print(f"MODEL EVALUATION ({self.extractor_name} + {self.classifier_name})")
        print("==================================================")
        print(f"- Train accuracy: {np.mean(train_preds == y_train) * 100:.2f}%")
        print(f"- Test accuracy: {np.mean(test_preds == y_test) * 100:.2f}%")
        print("==================================================\n")

    def _extract_feature_vector(self, image_path: str) -> np.ndarray:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not self.extractor or not self.classifier or not self.vocab or not self.vocab.is_fitted:
            raise ValueError("Pipeline is not configured or trained/loaded.")

        descriptors, color_hist = self._extract_descriptors_and_color(image_path)
        return self._fuse_features(descriptors, color_hist)

    def execute_pipeline(self, image_path: str) -> str:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        if self.classifier_name == "CNN":
            if not self.classifier.is_fitted:
                raise ValueError("CNN classifier service chưa được tải model.")
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Không thể đọc ảnh tại: {image_path}")
            return self.classifier.predict(img)

        feature_vector = self._extract_feature_vector(image_path)
        return self.classifier.predict(feature_vector)

    def execute_pipeline_with_proba(self, image_path: str) -> Tuple[str, dict]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        if self.classifier_name == "CNN":
            if not self.classifier.is_fitted:
                raise ValueError("CNN classifier service chưa được tải model.")
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Không thể đọc ảnh tại: {image_path}")
            pred = self.classifier.predict(img)
            probs = self.classifier.predict_proba(img)
            prob_dict = {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}
            return pred, prob_dict

        feature_vector = self._extract_feature_vector(image_path)
        pred = self.classifier.predict(feature_vector)
        probs = self.classifier.predict_proba(feature_vector)
        prob_dict = {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}
        return pred, prob_dict

    def predict(self, image_path: str) -> dict:
        feature_vector = self._extract_feature_vector(image_path)
        return self.classifier.predict_with_confidence(feature_vector)
