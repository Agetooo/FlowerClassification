import os
import time
from hub import ClassificationHub

def main():
    # Thư mục chứa tập dữ liệu huấn luyện
    dataset_dir = "it3160/flower-training"
    
    # Chọn ảnh mẫu để chạy thử (lấy ảnh đầu tiên của lớp daisy trong tập train)
    sample_image = os.path.join(dataset_dir, "daisy", "10140303196_b88d3d6cec.jpg")
    
    if not os.path.exists(sample_image):
        print(f"Không tìm thấy ảnh mẫu tại {sample_image}. Vui lòng kiểm tra lại đường dẫn.")
        return

    print("==================================================")
    print(" KHỞI TẠO ALGORITHM HUB & CHẠY CẤU HÌNH ORB + RF ")
    print("==================================================")
    
    # 1. Khởi tạo hub, chỉ định thư mục lưu mô hình là thư mục hiện tại
    hub = ClassificationHub(models_dir=".")
    
    # 2. Cấu hình pipeline: ORB + Random Forest (nhanh nhất)
    hub.set_extractor("ORB")
    hub.set_classifier("RandomForest")
    
    # 3. Kiểm tra xem đã có mô hình chưa, nếu chưa thì tự động train
    vocab_path = hub._get_vocab_path()
    clf_path = hub._get_classifier_path()
    
    if not os.path.exists(vocab_path) or not os.path.exists(clf_path):
        print("\nKhông tìm thấy mô hình đã huấn luyện của ORB + RF. Tiến hành huấn luyện...")
        start_time = time.time()
        # Huấn luyện trên tập dữ liệu hoa
        hub.train_pipeline(dataset_dir)
        print(f"Huấn luyện hoàn tất trong {time.time() - start_time:.2f} giây.")
    else:
        print("\nTải thành công mô hình ORB + RF đã lưu trước đó.")
        
    # 4. Chạy thử pipeline phân loại ảnh mẫu
    print(f"\nTiến hành dự đoán ảnh mẫu: '{sample_image}'")
    start_time = time.time()
    predicted_flower = hub.execute_pipeline(sample_image)
    duration = time.time() - start_time
    
    print(f"Kết quả dự đoán (ORB + RF): {predicted_flower}")
    print(f"Thời gian thực thi pipeline: {duration * 1000:.2f} ms")
    
    print("\n==================================================")
    print("   KIỂM TRA TÍNH TƯƠNG THÍCH NGƯỢC VỚI SVM CŨ   ")
    print("==================================================")
    
    # 1. Cấu hình bộ phân loại SVM (SVMService tự động tải best_svm_flower.joblib)
    hub.set_classifier("SVM")
    
    # 2. Chạy dự đoán ảnh mẫu bằng SVM cũ (sử dụng ảnh thô 32x32x3 phẳng làm phẳng)
    print(f"\nTiến hành dự đoán ảnh mẫu bằng SVM cũ: '{sample_image}'")
    start_time = time.time()
    predicted_flower_svm = hub.execute_pipeline(sample_image)
    duration_svm = time.time() - start_time
    
    print(f"Kết quả dự đoán (SVM cũ): {predicted_flower_svm}")
    print(f"Thời gian thực thi pipeline: {duration_svm * 1000:.2f} ms")

if __name__ == "__main__":
    main()
