import os
import cv2
import numpy as np
from PIL import Image
from typing import Optional, List, Tuple
from preprocess import preprocess_single_image
from services import (
    FeatureExtractorService, SIFTService, ORBService,
    ClassifierService, RandomForestService, XGBoostService, SVMService,
    VisualVocabulary
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
        self.classes = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

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
        
        # Khởi tạo VisualVocabulary tương ứng cho Extractor mới
        self.vocab = VisualVocabulary(n_clusters=100)
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
        else:
            raise ValueError(f"Không hỗ trợ thuật toán phân loại: {classifier_type}")
            
        self._auto_load_if_possible()

    def _get_vocab_path(self) -> str:
        return os.path.join(self.models_dir, f"vocab_{self.extractor_name.lower()}.joblib")

    def _get_classifier_path(self) -> str:
        if self.classifier_name == "SVM":
            # Tương thích ngược với file best_svm_flower.joblib ở thư mục chỉ định
            return os.path.join(self.models_dir, "best_svm_flower.joblib")
        return os.path.join(self.models_dir, f"model_{self.classifier_name.lower()}_{self.extractor_name.lower()}.joblib")

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

        # 2. Trích xuất đặc trưng từ tất cả các ảnh
        print("Đang trích xuất descriptors từ ảnh...")
        descriptors_list: List[np.ndarray] = []
        valid_indices: List[int] = []
        
        for idx, img_path in enumerate(image_paths):
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Sử dụng hàm làm sạch/tiền xử lý dữ liệu của preprocess.py
            cleaned_img = preprocess_single_image(img, targetedSize=(256, 256))
            descriptors = self.extractor.extract_features(cleaned_img)
            
            descriptors_list.append(descriptors)
            valid_indices.append(idx)
            
            if (idx + 1) % 500 == 0 or (idx + 1) == len(image_paths):
                print(f"Đã xử lý xong {idx + 1}/{len(image_paths)} ảnh.")

        # Lọc nhãn tương ứng với ảnh đọc thành công
        y = np.array([labels[i] for i in valid_indices])

        # 3. Huấn luyện Visual Vocabulary (BoVW)
        print("Đang huấn luyện Visual Vocabulary (K-Means)...")
        self.vocab.fit(descriptors_list)
        vocab_path = self._get_vocab_path()
        self.vocab.save(vocab_path)
        print(f"Đã lưu Visual Vocabulary thành công vào: {vocab_path}")

        # 4. Chuyển đổi sang histogram tần suất chuẩn hóa
        print("Đang biểu diễn đặc trưng ảnh dưới dạng BoVW Histograms...")
        X_hist = []
        for desc in descriptors_list:
            hist = self.vocab.transform(desc)
            X_hist.append(hist)
        X_hist = np.array(X_hist)

        # 5. Huấn luyện mô hình phân loại
        print(f"Đang huấn luyện bộ phân loại {self.classifier_name}...")
        if isinstance(self.classifier, RandomForestService):
            self.classifier.fit(X_hist, y, classes=self.classes)
        elif isinstance(self.classifier, XGBoostService):
            self.classifier.fit(X_hist, y, classes=self.classes)
            
        clf_path = self._get_classifier_path()
        self.classifier.save(clf_path)
        print(f"Đã lưu bộ phân loại thành công vào: {clf_path}")
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

        # Xử lý cho các pipeline SIFT/ORB + RF/XGBoost
        if not self.extractor or not self.classifier or not self.vocab.is_fitted:
            raise ValueError("Pipeline chưa được thiết lập đầy đủ hoặc chưa được huấn luyện/tải model.")

        # 1. Đọc ảnh
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Không thể đọc ảnh bằng OpenCV tại: {image_path}")

        # 2. Làm sạch & Tiền xử lý (tái sử dụng từ preprocess.py)
        preprocessed_img = preprocess_single_image(img, targetedSize=(256, 256))

        # 3. Trích xuất đặc trưng (SIFT/ORB)
        descriptors = self.extractor.extract_features(preprocessed_img)

        # 4. Tạo Visual Word/Histogram (BoVW) & Vector hóa
        feature_vector = self.vocab.transform(descriptors)

        # 5. Phân loại và trả về tên loài hoa
        return self.classifier.predict(feature_vector)
