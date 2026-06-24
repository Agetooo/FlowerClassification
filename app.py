import base64
import os
import threading

import cv2
from flask import Flask, jsonify, render_template, request

from hub import ClassificationHub
from preprocess import preprocess_single_image, extract_flower_mask


app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

hub = ClassificationHub(models_dir=".")

training_status = {
    "status": "idle",
    "progress": "",
    "error": "",
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Không tìm thấy file ảnh tải lên."}), 400

    file = request.files["image"]
    extractor_type = request.form.get("extractor", "ORB")
    classifier_type = request.form.get("classifier", "RandomForest")

    if file.filename == "":
        return jsonify({"error": "Tên tệp tin trống."}), 400

    temp_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(temp_path)

    try:
        # CNN không cần Feature Extractor (xử lý ảnh trực tiếp)
        raw_feature_clf = classifier_type.upper() == "CNN"

        if not raw_feature_clf:
            hub.set_extractor(extractor_type)
        hub.set_classifier(classifier_type)

        if raw_feature_clf:
            model_exists = os.path.exists(hub._get_classifier_path())
        else:
            model_exists = os.path.exists(hub._get_vocab_path()) and os.path.exists(hub._get_classifier_path())

        if not model_exists:
            return jsonify(
                {
                    "status": "not_trained",
                    "message": f"Mô hình cho cấu hình {extractor_type} + {classifier_type} chưa được huấn luyện.",
                }
            )

        img = cv2.imread(temp_path)
        if img is None:
            return jsonify({"error": "Không thể giải mã tệp ảnh đầu vào."}), 400

        preprocessed_img = preprocess_single_image(img, targetedSize=(256, 256))
        mask = extract_flower_mask(preprocessed_img)
        segmented_img = cv2.bitwise_and(preprocessed_img, preprocessed_img, mask=mask)

        _, orig_buffer = cv2.imencode(".jpg", img)
        orig_base64 = base64.b64encode(orig_buffer).decode("utf-8")

        _, prep_buffer = cv2.imencode(".jpg", segmented_img)
        prep_base64 = base64.b64encode(prep_buffer).decode("utf-8")

        predicted_class, probabilities = hub.execute_pipeline_with_proba(temp_path)
        confidence = max(probabilities.values()) if probabilities else 0.0
        print(f"[App Server] predicted: {predicted_class}, confidence: {confidence}")

        return jsonify(
            {
                "status": "success",
                "predicted_class": predicted_class,
                "confidence": confidence,
                "probabilities": probabilities,
                "original_image": f"data:image/jpeg;base64,{orig_base64}",
                "preprocessed_image": f"data:image/jpeg;base64,{prep_base64}",
                "config": {
                    "extractor": "None (64x64 pixels)" if classifier_type.upper() == "CNN" else extractor_type,
                    "classifier": classifier_type,
                },
            }
        )

    except Exception as exc:
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def train_worker(extractor_type: str, classifier_type: str):
    global training_status
    try:
        training_status["status"] = "training"
        training_status["progress"] = "Khoi chay luong huan luyen nen..."
        training_status["error"] = ""

        hub.set_extractor(extractor_type)
        hub.set_classifier(classifier_type)

        training_status["progress"] = "Dang trich xuat dac trung HOG va tinh BoVW..."
        hub.train_pipeline("flower-training")

        training_status["status"] = "completed"
        training_status["progress"] = f"Huan luyen thanh cong cau hinh {extractor_type} + {classifier_type}!"
    except Exception as exc:
        training_status["status"] = "failed"
        training_status["error"] = str(exc)
        training_status["progress"] = f"Huan luyen that bai: {exc}"


@app.route("/train", methods=["POST"])
def train():
    global training_status
    if training_status["status"] == "training":
        return jsonify({"status": "error", "message": "Mot tien trinh huan luyen dang hoat dong."}), 400

    extractor_type = request.form.get("extractor", "ORB")
    classifier_type = request.form.get("classifier", "RandomForest")

    training_status["status"] = "training"
    training_status["progress"] = "Dang chuan bi du lieu..."
    training_status["error"] = ""

    thread = threading.Thread(target=train_worker, args=(extractor_type, classifier_type))
    thread.start()

    return jsonify({"status": "started", "message": "Tien trinh huan luyen nen da bat dau."})


@app.route("/train_status", methods=["GET"])
def train_status():
    return jsonify(training_status)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
