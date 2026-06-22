import os
import time
import threading
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from hub import ClassificationHub
from preprocess import preprocess_single_image

app = Flask(__name__)

# Thư mục lưu tệp ảnh tạm thời khi tải lên
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Khởi tạo trung tâm điều khiển Algorithm Hub
hub = ClassificationHub(models_dir=".")

# Lưu trạng thái tiến trình huấn luyện toàn cục (cho luồng nền)
training_status = {
    "status": "idle",  # idle, training, completed, failed
    "progress": "",
    "error": ""
}

def get_probability_dict(hub_instance: ClassificationHub, feature_vector: np.ndarray, classifier_name: str) -> dict:
    """
    Trả về từ điển xác suất dự báo của từng loài hoa.
    """
    try:
        if classifier_name.upper() == "SVM":
            # SVMService tự tính toán softmax của decision_function
            probs = hub_instance.classifier.predict_proba(feature_vector)
        else:
            probs = hub_instance.classifier.predict_proba(feature_vector)
        
        return {hub_instance.classes[i]: float(probs[i]) for i in range(len(hub_instance.classes))}
    except Exception as e:
        print("[App] Lỗi tính xác suất lớp:", e)
        return {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "Không tìm thấy file ảnh tải lên."}), 400
        
    file = request.files['image']
    extractor_type = request.form.get('extractor', 'ORB')
    classifier_type = request.form.get('classifier', 'RandomForest')
    
    if file.filename == '':
        return jsonify({"error": "Tên tệp tin trống."}), 400
        
    # Lưu ảnh tạm thời xuống đĩa để xử lý
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(temp_path)
    
    try:
        # SVM và Softmax tự xử lý đặc trưng riêng, không cần Feature Extractor (SIFT/ORB)
        raw_feature_clf = classifier_type.upper() in ("SVM", "CNN")

        # Cấu hình lại Extractor và Classifier
        if not raw_feature_clf:
            hub.set_extractor(extractor_type)
        hub.set_classifier(classifier_type)

        # Kiểm tra xem mô hình của cấu hình này đã được huấn luyện chưa
        if raw_feature_clf:
            model_exists = os.path.exists(hub._get_classifier_path())
        else:
            model_exists = os.path.exists(hub._get_vocab_path()) and os.path.exists(hub._get_classifier_path())
            
        if not model_exists:
            return jsonify({
                "status": "not_trained",
                "message": f"Mô hình cho cấu hình {extractor_type} + {classifier_type} chưa được huấn luyện."
            })
            
        # Đọc ảnh thô
        img = cv2.imread(temp_path)
        if img is None:
            return jsonify({"error": "Không thể giải mã tệp ảnh đầu vào."}), 400
            
        # Áp dụng hàm tiền xử lý (resize, Gaussian blur) từ preprocess.py
        preprocessed_img = preprocess_single_image(img, targetedSize=(256, 256))
        
        # Mã hóa base64 ảnh gốc và ảnh đã tiền xử lý để hiển thị so sánh trên UI
        _, orig_buffer = cv2.imencode('.jpg', img)
        orig_base64 = base64.b64encode(orig_buffer).decode('utf-8')
        
        _, prep_buffer = cv2.imencode('.jpg', preprocessed_img)
        prep_base64 = base64.b64encode(prep_buffer).decode('utf-8')
        
        # Dự đoán nhãn lớp và tính xác suất phân phối bằng Hub
        predicted_class, probabilities = hub.execute_pipeline_with_proba(temp_path)
        print(f"[App Server] predicted: {predicted_class}, probabilities: {probabilities}")

        return jsonify({
            "status": "success",
            "predicted_class": predicted_class,
            "probabilities": probabilities,
            "original_image": f"data:image/jpeg;base64,{orig_base64}",
            "preprocessed_image": f"data:image/jpeg;base64,{prep_base64}",
            "config": {
                "extractor": (
                    "None (Raw Pixels)" if classifier_type.upper() == "SVM"
                    else "None (64x64 pixels)" if classifier_type.upper() == "CNN"
                    else extractor_type
                ),
                "classifier": classifier_type
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        # Dọn dẹp tệp ảnh tạm thời
        if os.path.exists(temp_path):
            os.remove(temp_path)


def train_worker(extractor_type: str, classifier_type: str):
    """
    Hàm nền chạy huấn luyện độc lập tránh nghẽn thread của Flask Server.
    """
    global training_status
    try:
        training_status["status"] = "training"
        training_status["progress"] = "Khởi chạy luồng huấn luyện nền..."
        training_status["error"] = ""
        
        # Cấu hình Hub
        hub.set_extractor(extractor_type)
        hub.set_classifier(classifier_type)
        
        training_status["progress"] = "Đang trích xuất đặc trưng và tính BoVW..."
        
        # Thực thi pipeline huấn luyện
        hub.train_pipeline("flower-training")
        
        training_status["status"] = "completed"
        training_status["progress"] = f"Huấn luyện thành công cấu hình {extractor_type} + {classifier_type}!"
    except Exception as e:
        training_status["status"] = "failed"
        training_status["error"] = str(e)
        training_status["progress"] = f"Huấn luyện thất bại: {str(e)}"


@app.route('/train', methods=['POST'])
def train():
    global training_status
    if training_status["status"] == "training":
        return jsonify({"status": "error", "message": "Một tiến trình huấn luyện đang hoạt động."}), 400
        
    extractor_type = request.form.get('extractor', 'ORB')
    classifier_type = request.form.get('classifier', 'RandomForest')
    
    # Reset và chạy luồng nền
    training_status["status"] = "training"
    training_status["progress"] = "Đang chuẩn bị dữ liệu..."
    training_status["error"] = ""
    
    thread = threading.Thread(target=train_worker, args=(extractor_type, classifier_type))
    thread.start()
    
    return jsonify({"status": "started", "message": "Tiến trình huấn luyện nền đã bắt đầu."})


@app.route('/train_status', methods=['GET'])
def train_status():
    global training_status
    return jsonify(training_status)


if __name__ == '__main__':
    # Chạy trên localhost cổng 5000
    app.run(debug=True, host='127.0.0.1', port=5000)
