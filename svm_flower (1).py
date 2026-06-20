"""
SVM Flower Image Classification System
Dataset structure:
flower-training/
    daisy/
    dandelion/
    rose/
    sunflower/
    tulip/

Pipeline:
Load images -> Resize -> Flatten -> Split train/validation/test -> Normalize -> Train SVM
-> Tune C -> Evaluate -> Save best model -> Predict single image.
"""

import os
import joblib
import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CLASSES = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
IMG_SIZE = (32, 32)
DATASET_DIR = "flower-training"
RANDOM_STATE = 42


def load_flower_data(data_dir=DATASET_DIR, img_size=IMG_SIZE):
    """Load flower images from folders and convert them to RGB arrays."""
    X = []
    y = []

    for label, cls in enumerate(CLASSES):
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError(f"Folder not found: {cls_dir}")

        for filename in os.listdir(cls_dir):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                continue
            img_path = os.path.join(cls_dir, filename)
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(img_size)
                X.append(np.array(img))
                y.append(label)
            except Exception as e:
                print(f"Skipping {img_path}: {e}")

    X = np.array(X).astype("float64")
    y = np.array(y)

    np.random.seed(RANDOM_STATE)
    idxs = np.random.permutation(len(X))
    return X[idxs], y[idxs]


def preprocess_data(X_train_raw, X_val_raw, X_test_raw):
    """Flatten images, subtract mean image, then standardize features."""
    X_train = X_train_raw.reshape(X_train_raw.shape[0], -1)
    X_val = X_val_raw.reshape(X_val_raw.shape[0], -1)
    X_test = X_test_raw.reshape(X_test_raw.shape[0], -1)

    mean_image = np.mean(X_train, axis=0)
    X_train = X_train - mean_image
    X_val = X_val - mean_image
    X_test = X_test - mean_image

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, mean_image, scaler


def train_and_tune_svm(X_train, y_train, X_val, y_val):
    """Train several Linear SVM models and choose the best by validation accuracy."""
    C_values = [0.001, 0.01, 0.1, 1, 10]
    results = {}
    best_val = -1
    best_svm = None
    best_params = None

    for C in C_values:
        svm = LinearSVC(
            C=C,
            class_weight="balanced",
            max_iter=10000,
            random_state=RANDOM_STATE,
        )
        svm.fit(X_train, y_train)

        train_pred = svm.predict(X_train)
        val_pred = svm.predict(X_val)
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)

        results[C] = {"train_accuracy": train_acc, "validation_accuracy": val_acc}
        print(f"C={C:<6} | train acc={train_acc:.4f} | val acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            best_svm = svm
            best_params = {"C": C}

    print("\nBest validation accuracy:", best_val)
    print("Best parameters:", best_params)
    return best_svm, best_params, results


def evaluate_model(model, X_test, y_test):
    """Evaluate selected SVM on test set."""
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print("\nSVM test accuracy:", test_accuracy)
    print("\nClassification report:")
    print(classification_report(y_test, y_test_pred, target_names=CLASSES))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_test_pred))

    return test_accuracy, y_test_pred


def save_model(model, best_params, mean_image, scaler, results, path="best_svm_flower.joblib"):
    """Save model and preprocessing objects needed for future prediction."""
    joblib.dump(
        {
            "model": model,
            "classes": CLASSES,
            "img_size": IMG_SIZE,
            "mean_image": mean_image,
            "scaler": scaler,
            "best_params": best_params,
            "results": results,
            "model_type": "LinearSVC",
        },
        path,
    )
    print(f"\nSaved: {path}")


def predict_single_image(image_path, model_package_path="best_svm_flower.joblib"):
    """Predict one image using saved SVM model package."""
    package = joblib.load(model_package_path)
    model = package["model"]
    classes = package["classes"]
    img_size = package["img_size"]
    mean_image = package["mean_image"]
    scaler = package["scaler"]

    img = Image.open(image_path).convert("RGB")
    img = img.resize(img_size)
    arr = np.array(img).astype("float64").reshape(1, -1)
    arr = arr - mean_image
    arr = scaler.transform(arr)

    pred = model.predict(arr)[0]
    scores = model.decision_function(arr)[0]

    print("Predicted class:", classes[pred])
    print("\nDecision scores:")
    for cls, score in zip(classes, scores):
        print(f"{cls}: {score:.4f}")

    return classes[pred], scores


if __name__ == "__main__":
    print("Loading dataset...")
    X, y = load_flower_data()
    print("Dataset shape:", X.shape)
    print("Labels shape:", y.shape)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_raw, y_train, test_size=0.1, random_state=RANDOM_STATE, stratify=y_train
    )

    print("\nData split:")
    print("Train:", X_train_raw.shape, y_train.shape)
    print("Validation:", X_val_raw.shape, y_val.shape)
    print("Test:", X_test_raw.shape, y_test.shape)

    print("\nPreprocessing...")
    X_train, X_val, X_test, mean_image, scaler = preprocess_data(
        X_train_raw, X_val_raw, X_test_raw
    )
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)

    print("\nTraining and tuning SVM...")
    best_svm, best_params, results = train_and_tune_svm(X_train, y_train, X_val, y_val)

    print("\nEvaluating best SVM on test set...")
    evaluate_model(best_svm, X_test, y_test)

    save_model(best_svm, best_params, mean_image, scaler, results)
