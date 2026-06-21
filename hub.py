import os
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from typing import Optional, List, Tuple
from preprocess import preprocess_single_image, extract_flower_mask
from services import (
    FeatureExtractorService, SIFTService, ORBService,
    ClassifierService, RandomForestService, XGBoostService, SVMService, SoftmaxService,
    VisualVocabulary, extract_hsv_histogram
)

class ClassificationHub:
    """
    Central Controller (Algorithm Hub) sử dụng Strategy và Factory Pattern
    để điều phối việc thiết lập, huấn luyện và thực thi các pipeline phân loại hoa.
    """
    def __init__(self, models_dir: str = "."):
        self.models_dir = models_dir
        self.extractor: Optional[FeatureExtractorService] = None
        self.classifier: Optional[ClassifierService] = None
        self.vocab: Optional[VisualVocabulary] = None
        self.extractor_name: str = ""
        self.classifier_name: str = ""
        
        # Danh sách loài hoa đồng nhất với nhãn phân lớp
        self.classes = ["bellflower", "daisy", "dandelion", "lotus", "rose", "sunflower", "tulip"]

    def set_extractor(self, extractor_type: str) -> None:
        """
        Khởi tạo Feature Extractor Service dựa trên Factory Pattern.
        
        Args:
            extractor_type (str): 'SIFT' hoặc 'ORB'
        """
        name = extractor_type.upper()
        if name == "SIFT":
            self.extractor = SIFTService()
            self.extractor_name = "SIFT"
        elif name == "ORB":
            self.extractor = ORBService()
            self.extractor_name = "ORB"
        else:
            raise ValueError(f"Không hỗ trợ thuật toán trích xuất: {extractor_type}")
        
        # Khởi tạo VisualVocabulary tương ứng cho Extractor mới với K=300
        self.vocab = VisualVocabulary(n_clusters=300)
        self._auto_load_if_possible()

    def set_classifier(self, classifier_type: str) -> None:
        """
        Khởi tạo Classifier Service dựa trên Factory Pattern.
        
        Args:
            classifier_type (str): 'RandomForest', 'XGBoost' hoặc 'SVM'
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
        elif name == "SOFTMAX":
            self.classifier = SoftmaxService()
            self.classifier_name = "Softmax"
        else:
            raise ValueError(f"Không hỗ trợ thuật toán phân loại: {classifier_type}")
            
        self._auto_load_if_possible()

    def _get_vocab_path(self) -> str:
        return os.path.join(self.models_dir, f"vocab_{self.extractor_name.lower()}_opt.joblib")

    def _get_softmax_dir(self) -> str:
        # Thư mục chứa model Softmax 7 lớp (Hu Moments + HSV Histogram)
        return os.path.join(self.models_dir, "it3160", "HuHSVHistSoftmaxModel")

    def _get_classifier_path(self) -> str:
        if self.classifier_name == "SVM":
            # Tương thích ngược với file best_svm_flower.joblib ở thư mục chỉ định
            return os.path.join(self.models_dir, "best_svm_flower.joblib")
        if self.classifier_name == "Softmax":
            # Dùng W_weights.npy làm "mốc" để app.py kiểm tra model đã tồn tại chưa
            return os.path.join(self._get_softmax_dir(), "W_weights.npy")
        return os.path.join(self.models_dir, f"model_{self.classifier_name.lower()}_{self.extractor_name.lower()}_opt.joblib")

    def _auto_load_if_possible(self) -> None:
        """
        Tự động kiểm tra và tải các tệp mô hình đã huấn luyện nếu có sẵn trên đĩa.
        """
        # SVM chỉ cần tải model
        if self.classifier_name == "SVM":
            path = self._get_classifier_path()
            if os.path.exists(path):
                self.classifier.load(path)
                print(f"[Hub] Đã tự động tải mô hình SVM tương thích ngược từ: {path}")
            return

        # Softmax chỉ cần tải thư mục model (W, b, classes, X_features)
        if self.classifier_name == "Softmax":
            softmax_dir = self._get_softmax_dir()
            if os.path.exists(os.path.join(softmax_dir, "W_weights.npy")):
                self.classifier.load(softmax_dir)
                print(f"[Hub] Đã tự động tải mô hình Softmax 7 lớp từ: {softmax_dir}")
            return

        # SIFT/ORB + RF/XGBoost cần cả extractor, vocab và classifier
        if self.extractor_name and self.classifier_name:
            vocab_path = self._get_vocab_path()
            clf_path = self._get_classifier_path()
            
            if os.path.exists(vocab_path) and os.path.exists(clf_path):
                self.vocab.load(vocab_path)
                self.classifier.load(clf_path)
                print(f"[Hub] Đã tự động tải thành công pipeline: {self.extractor_name} + {self.classifier_name}")

    def train_pipeline(self, dataset_dir: str) -> None:
        """
        Huấn luyện đầy đủ cho cấu hình pipeline hiện tại (trừ SVM).
        
        Args:
            dataset_dir (str): Thư mục chứa các loài hoa (flower-training)
        """
        if self.classifier_name == "SVM":
            print("[Hub] SVM sử dụng mô hình pre-trained tương thích ngược, không cần train.")
            return

        if self.classifier_name == "Softmax":
            print("[Hub] Softmax sử dụng mô hình pre-trained (Hu+HSV), không cần train.")
            return

        if not self.extractor_name or not self.classifier_name:
            raise ValueError("Hãy thiết lập đầy đủ Extractor và Classifier trước khi train.")

        print(f"\n--- Bắt đầu huấn luyện Pipeline: {self.extractor_name} + {self.classifier_name} ---")
        
        # 1. Đọc danh sách ảnh từ tập dữ liệu
        image_paths: List[str] = []
        labels: List[int] = []
        
        for label_idx, cls in enumerate(self.classes):
            cls_dir = os.path.join(dataset_dir, cls)
            if not os.path.exists(cls_dir):
                # Thử tìm trong it3160/flower-training nếu không thấy ở đường dẫn gốc
                cls_dir = os.path.join("it3160", dataset_dir, cls)
                if not os.path.exists(cls_dir):
                    print(f"Bỏ qua thư mục lớp không tồn tại: {cls_dir}")
                    continue
            
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    image_paths.append(os.path.join(cls_dir, img_name))
                    labels.append(label_idx)

        if not image_paths:
            raise ValueError(f"Không tìm thấy ảnh huấn luyện nào trong thư mục: {dataset_dir}")
            
        print(f"Tổng số ảnh thu thập: {len(image_paths)}")

        # 2. Trích xuất đặc trưng (BoVW descriptors & Color Histograms) từ tất cả các ảnh
        print("Đang trích xuất đặc trưng kết cấu và màu sắc từ tập dữ liệu...")
        descriptors_list: List[np.ndarray] = []
        color_hists_list: List[np.ndarray] = []
        valid_indices: List[int] = []
        
        for idx, img_path in enumerate(image_paths):
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Sử dụng hàm làm sạch/tiền xử lý dữ liệu của preprocess.py
            cleaned_img = preprocess_single_image(img, targetedSize=(256, 256))
            # Tạo mặt nạ khử nhiễu nền xanh lá cây
            mask = extract_flower_mask(cleaned_img)
            
            # Trích xuất descriptors cục bộ chỉ trên vùng hoa
            descriptors = self.extractor.extract_features(cleaned_img, mask=mask)
            # Trích xuất histogram màu sắc HSV chỉ trên vùng hoa
            color_hist = extract_hsv_histogram(cleaned_img, mask=mask)
            
            descriptors_list.append(descriptors)
            color_hists_list.append(color_hist)
            valid_indices.append(idx)
            
            if (idx + 1) % 500 == 0 or (idx + 1) == len(image_paths):
                print(f"Đã xử lý xong {idx + 1}/{len(image_paths)} ảnh.")

        # Lọc nhãn tương ứng với ảnh đọc thành công
        y = np.array([labels[i] for i in valid_indices])

        # Chia tập dữ liệu thành 80% Train và 20% Test (Stratified)
        indices = np.arange(len(y))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=y, random_state=42)
        print(f"[Hub] Phân chia tập dữ liệu: {len(train_idx)} ảnh huấn luyện, {len(test_idx)} ảnh kiểm thử.")

        # 3. Huấn luyện Visual Vocabulary (BoVW) trên tập Train
        print("Đang huấn luyện Visual Vocabulary (K-Means) trên tập huấn luyện...")
        train_descriptors = [descriptors_list[i] for i in train_idx]
        self.vocab.fit(train_descriptors)
        vocab_path = self._get_vocab_path()
        self.vocab.save(vocab_path)
        print(f"Đã lưu Visual Vocabulary thành công vào: {vocab_path}")

        # 4. Biểu diễn đặc trưng ảnh dưới dạng Fused Color-Texture Vectors cho cả Train và Test
        print("Đang biểu diễn đặc trưng ảnh dưới dạng Fused Color-Texture Vectors...")
        
        X_train = []
        for i in train_idx:
            hist = self.vocab.transform(descriptors_list[i])
            color_hist = color_hists_list[i]
            hist_l2 = hist / (np.linalg.norm(hist, ord=2) + 1e-6)
            color_l2 = color_hist / (np.linalg.norm(color_hist, ord=2) + 1e-6)
            fused_vector = np.hstack([hist_l2, 0.5 * color_l2])
            fused_vector /= (np.linalg.norm(fused_vector, ord=2) + 1e-6)
            X_train.append(fused_vector)
        X_train = np.array(X_train)
        y_train = y[train_idx]

        X_test = []
        for i in test_idx:
            hist = self.vocab.transform(descriptors_list[i])
            color_hist = color_hists_list[i]
            hist_l2 = hist / (np.linalg.norm(hist, ord=2) + 1e-6)
            color_l2 = color_hist / (np.linalg.norm(color_hist, ord=2) + 1e-6)
            fused_vector = np.hstack([hist_l2, 0.5 * color_l2])
            fused_vector /= (np.linalg.norm(fused_vector, ord=2) + 1e-6)
            X_test.append(fused_vector)
        X_test = np.array(X_test)
        y_test = y[test_idx]

        # 5. Huấn luyện mô hình phân loại trên tập Train
        print(f"Đang huấn luyện bộ phân loại {self.classifier_name}...")
        self.classifier.fit(X_train, y_train, classes=self.classes)
            
        clf_path = self._get_classifier_path()
        self.classifier.save(clf_path)
        print(f"Đã lưu bộ phân loại thành công vào: {clf_path}")

        # 6. Đánh giá tính tổng quát (Generalization Evaluation)
        train_preds = [self.classifier.predict(x) for x in X_train]
        train_pred_indices = [self.classes.index(p) for p in train_preds]
        train_accuracy = np.mean(np.array(train_pred_indices) == y_train)

        test_preds = [self.classifier.predict(x) for x in X_test]
        test_pred_indices = [self.classes.index(p) for p in test_preds]
        test_accuracy = np.mean(np.array(test_pred_indices) == y_test)

        print(f"\n==================================================")
        print(f"   ĐÁNH GIÁ HIỆU NĂNG MÔ HÌNH ({self.extractor_name} + {self.classifier_name})")
        print(f"==================================================")
        print(f"- Độ chính xác trên tập Train: {train_accuracy * 100:.2f}%")
        print(f"- Độ chính xác trên tập Test: {test_accuracy * 100:.2f}%")
        print(f"==================================================\n")
        print("--- Hoàn tất huấn luyện pipeline! ---\n")

    def execute_pipeline(self, image_path: str) -> str:
        """
        Thực hiện toàn bộ vòng đời phân loại cho một ảnh:
        Đọc ảnh -> Làm sạch/Tiền xử lý -> Trích xuất đặc trưng -> Tạo Visual Word/Histogram (BoVW) 
        -> Vector hóa -> Phân loại -> Trả về kết quả loài hoa.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Không tìm thấy ảnh tại: {image_path}")

        # Xử lý riêng biệt cho SVM (Backward compatibility)
        if self.classifier_name == "SVM":
            if not self.classifier.is_fitted:
                raise ValueError("SVM classifier service chưa được tải model.")
            
            # Đọc ảnh theo đúng cách của svm_flower.py gốc
            img = Image.open(image_path).convert("RGB")
            img = img.resize(self.classifier.img_size)
            # Tạo vector phẳng 3072 chiều
            arr = np.array(img).astype("float64").reshape(-1)
            
            # Predict
            return self.classifier.predict(arr)

        # Xử lý riêng biệt cho Softmax (Hu Moments + HSV Histogram, 7 lớp)
        if self.classifier_name == "Softmax":
            if not self.classifier.is_fitted:
                raise ValueError("Softmax classifier service chưa được tải model.")
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Không thể đọc ảnh bằng OpenCV tại: {image_path}")
            feat = self.classifier.extract_features(img)
            return self.classifier.predict(feat)

        # Xử lý cho các pipeline SIFT/ORB + RF/XGBoost
        if not self.extractor or not self.classifier or not self.vocab.is_fitted:
            raise ValueError("Pipeline chưa được thiết lập đầy đủ hoặc chưa được huấn luyện/tải model.")

        # 1. Đọc ảnh
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Không thể đọc ảnh bằng OpenCV tại: {image_path}")

        # 2. Làm sạch & Tiền xử lý (tái sử dụng từ preprocess.py)
        preprocessed_img = preprocess_single_image(img, targetedSize=(256, 256))
        
        # 3. Tạo mặt nạ loại bỏ nền xanh lá cây
        mask = extract_flower_mask(preprocessed_img)

        # 4. Trích xuất đặc trưng (SIFT/ORB) và đặc trưng màu sắc HSV chỉ trên vùng hoa
        descriptors = self.extractor.extract_features(preprocessed_img, mask=mask)
        color_hist = extract_hsv_histogram(preprocessed_img, mask=mask)

        # 5. Tạo Visual Word/Histogram (BoVW) & Kết hợp đặc trưng màu sắc
        hist = self.vocab.transform(descriptors)
        
        # Đồng chuẩn hóa L2 để cả hai đặc trưng có cùng thang đo
        hist_l2 = hist / (np.linalg.norm(hist, ord=2) + 1e-6)
        color_l2 = color_hist / (np.linalg.norm(color_hist, ord=2) + 1e-6)
        
        # Kết hợp đặc trưng BoVW (300 chiều) và Color Histogram (128 chiều) với trọng số cho màu sắc
        feature_vector = np.hstack([hist_l2, 0.5 * color_l2])
        # Chuẩn hóa L2 toàn bộ vector đã kết hợp
        feature_vector /= (np.linalg.norm(feature_vector, ord=2) + 1e-6)

        # 6. Phân loại và trả về tên loài hoa
        return self.classifier.predict(feature_vector)

    def execute_pipeline_with_proba(self, image_path: str) -> Tuple[str, dict]:
        """
        Thực hiện toàn bộ vòng đời phân loại cho một ảnh và trả về cả lớp dự báo lẫn
        từ điển xác suất của các loài hoa.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Không tìm thấy ảnh tại: {image_path}")

        # Xử lý riêng biệt cho SVM (Backward compatibility)
        if self.classifier_name == "SVM":
            if not self.classifier.is_fitted:
                raise ValueError("SVM classifier service chưa được tải model.")
            
            img = Image.open(image_path).convert("RGB")
            img = img.resize(self.classifier.img_size)
            arr = np.array(img).astype("float64").reshape(-1)
            
            pred = self.classifier.predict(arr)
            probs = self.classifier.predict_proba(arr)
            prob_dict = {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}
            return pred, prob_dict

        # Xử lý riêng biệt cho Softmax (Hu Moments + HSV Histogram, 7 lớp)
        if self.classifier_name == "Softmax":
            if not self.classifier.is_fitted:
                raise ValueError("Softmax classifier service chưa được tải model.")
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Không thể đọc ảnh bằng OpenCV tại: {image_path}")
            feat = self.classifier.extract_features(img)
            pred = self.classifier.predict(feat)
            probs = self.classifier.predict_proba(feat)
            prob_dict = {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}
            return pred, prob_dict

        # Xử lý cho các pipeline SIFT/ORB + RF/XGBoost
        if not self.extractor or not self.classifier or not self.vocab.is_fitted:
            raise ValueError("Pipeline chưa được thiết lập đầy đủ hoặc chưa được huấn luyện/tải model.")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Không thể đọc ảnh bằng OpenCV tại: {image_path}")

        preprocessed_img = preprocess_single_image(img, targetedSize=(256, 256))
        mask = extract_flower_mask(preprocessed_img)
        descriptors = self.extractor.extract_features(preprocessed_img, mask=mask)
        color_hist = extract_hsv_histogram(preprocessed_img, mask=mask)

        hist = self.vocab.transform(descriptors)
        hist_l2 = hist / (np.linalg.norm(hist, ord=2) + 1e-6)
        color_l2 = color_hist / (np.linalg.norm(color_hist, ord=2) + 1e-6)
        feature_vector = np.hstack([hist_l2, 0.5 * color_l2])
        feature_vector /= (np.linalg.norm(feature_vector, ord=2) + 1e-6)

        pred = self.classifier.predict(feature_vector)
        probs = self.classifier.predict_proba(feature_vector)
        prob_dict = {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}
        return pred, prob_dict
