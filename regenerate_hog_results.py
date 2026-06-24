import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from skimage.feature import hog
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, learning_curve, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CACHE_PATH = PROJECT_ROOT / "scratch" / "hog_flowers_features_128.npz"

CLASS_NAMES = ["Bellflower", "Daisy", "Dandelion", "Lotus", "Rose", "Sunflower", "Tulip"]
CLASS_KEYS = ["bellflower", "daisy", "dandelion", "lotus", "rose", "sunflower", "tulip"]
CLASS_ALIASES = {"tulip": ["tulip", "tulips"], "tulips": ["tulip", "tulips"]}

IMG_SIZE = (128, 128)
TEST_SIZE = 0.20
RANDOM_STATE = 42
LEARNING_CURVE_MAX_SAMPLES = 3500
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_class_dirs(root: Path):
    if not root.exists() or not root.is_dir():
        return None

    children = {child.name.lower(): child for child in root.iterdir() if child.is_dir()}
    class_dirs = {}
    for key in CLASS_KEYS:
        aliases = CLASS_ALIASES.get(key, [key])
        match = next((children[alias] for alias in aliases if alias in children), None)
        if match is None:
            return None
        class_dirs[key] = match
    return class_dirs


def find_dataset_dir():
    candidates = [
        PROJECT_ROOT / "flowers" / "flowers",
        PROJECT_ROOT / "flowers",
        PROJECT_ROOT / "flower-training",
        PROJECT_ROOT / "dataset",
        PROJECT_ROOT / "data",
    ]
    for candidate in candidates:
        class_dirs = resolve_class_dirs(candidate)
        if class_dirs is not None:
            return candidate, class_dirs

    raise FileNotFoundError("Cannot find dataset folder with all 7 flower classes.")


def list_images(class_dirs):
    image_paths, labels = [], []
    for label, key in enumerate(CLASS_KEYS):
        files = sorted(
            path
            for path in class_dirs[key].iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        )
        image_paths.extend(files)
        labels.extend([label] * len(files))
    return image_paths, np.array(labels, dtype=np.int64)


def extract_hog_feature(image_path: Path):
    image_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    image = cv2.resize(image, IMG_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    )
    return feature.astype(np.float32)


def load_features(image_paths):
    if CACHE_PATH.exists():
        cached = np.load(CACHE_PATH)
        if int(cached["num_paths"]) == len(image_paths):
            print(f"Loading cached HOG features: {CACHE_PATH}")
            return cached["X"], cached["valid_indices"]

    features, valid_indices = [], []
    for idx, image_path in enumerate(image_paths):
        try:
            features.append(extract_hog_feature(image_path))
            valid_indices.append(idx)
        except ValueError as exc:
            print(f"Skipping {image_path}: {exc}")

        if (idx + 1) % 500 == 0 or idx + 1 == len(image_paths):
            print(f"Extracted HOG: {idx + 1}/{len(image_paths)}")

    X = np.vstack(features)
    valid_indices = np.array(valid_indices, dtype=np.int64)
    CACHE_PATH.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        CACHE_PATH,
        X=X,
        valid_indices=valid_indices,
        num_paths=np.array(len(image_paths), dtype=np.int64),
    )
    print(f"Cached HOG features: {CACHE_PATH}")
    return X, valid_indices


def build_models():
    return {
        "HOG + SVM": Pipeline(
            [
                (
                    "classifier",
                    LinearSVC(
                        C=0.003,
                        class_weight="balanced",
                        dual="auto",
                        max_iter=5000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "HOG + KNN": Pipeline(
            [
                (
                    "classifier",
                    KNeighborsClassifier(
                        n_neighbors=3,
                        weights="uniform",
                        metric="manhattan",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def plot_confusion_matrix_png(y_true, y_pred, title, output_name):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASS_NAMES)))
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar=False,
    )
    plt.title(title, fontsize=15, weight="bold")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / output_name, dpi=200)
    plt.close()


def learning_curve_data(X, y):
    if LEARNING_CURVE_MAX_SAMPLES is None or LEARNING_CURVE_MAX_SAMPLES >= len(y):
        return X, y

    _, X_lc, _, y_lc = train_test_split(
        X,
        y,
        test_size=LEARNING_CURVE_MAX_SAMPLES,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_lc, y_lc


def plot_learning_curve_png(model, title, output_name, X, y):
    X_lc, y_lc = learning_curve_data(X, y)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    train_sizes, train_scores, valid_scores = learning_curve(
        model,
        X_lc,
        y_lc,
        cv=cv,
        scoring="accuracy",
        train_sizes=np.linspace(0.2, 1.0, 5),
        n_jobs=1,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    valid_mean = valid_scores.mean(axis=1)
    valid_std = valid_scores.std(axis=1)

    plt.figure(figsize=(9, 6))
    plt.plot(train_sizes, train_mean, marker="o", label="Train accuracy")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.18)
    plt.plot(train_sizes, valid_mean, marker="s", label="CV accuracy")
    plt.fill_between(train_sizes, valid_mean - valid_std, valid_mean + valid_std, alpha=0.18)
    plt.title(title, fontsize=15, weight="bold")
    plt.xlabel("Training samples")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / output_name, dpi=200)
    plt.close()

    return train_sizes, train_mean, valid_mean


def plot_accuracy_comparison(metrics):
    names = list(metrics)
    accuracies = [metrics[name]["accuracy"] for name in names]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, accuracies, color=["#2563eb", "#16a34a"])
    plt.ylim(0, 1)
    plt.ylabel("Test accuracy")
    plt.title("Accuracy Comparison - HOG + SVM vs HOG + KNN", fontsize=15, weight="bold")
    for bar, accuracy in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            accuracy + 0.02,
            f"{accuracy:.3f}",
            ha="center",
            va="bottom",
            weight="bold",
        )
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "hog_svm_knn_flowers_accuracy_comparison.png", dpi=200)
    plt.close()


def main():
    dataset_dir, class_dirs = find_dataset_dir()
    print(f"Dataset: {dataset_dir}")
    for name, key in zip(CLASS_NAMES, CLASS_KEYS):
        count = len([p for p in class_dirs[key].iterdir() if p.suffix.lower() in IMAGE_EXTS])
        print(f"- {name}: {count}")

    image_paths, y_all = list_images(class_dirs)
    X_all, valid_indices = load_features(image_paths)
    y_all = y_all[valid_indices]

    X_train, X_test, y_train, y_test = train_test_split(
        X_all,
        y_all,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_all,
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    models = build_models()
    predictions = {}
    metrics = {}
    report_lines = [f"Dataset: {dataset_dir}", f"Features: {X_all.shape}", ""]

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        predictions[name] = y_test_pred

        accuracy = accuracy_score(y_test, y_test_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_test_pred, average="weighted", zero_division=0
        )
        train_accuracy = accuracy_score(y_train, y_train_pred)
        metrics[name] = {
            "train_accuracy": train_accuracy,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        print(f"{name}: train={train_accuracy:.4f}, test={accuracy:.4f}, f1={f1:.4f}")
        report_lines.extend(
            [
                name,
                f"Train accuracy: {train_accuracy:.4f}",
                f"Test accuracy: {accuracy:.4f}",
                f"Precision: {precision:.4f}",
                f"Recall: {recall:.4f}",
                f"F1-score: {f1:.4f}",
                classification_report(y_test, y_test_pred, target_names=CLASS_NAMES, zero_division=0),
                "",
            ]
        )

    plot_confusion_matrix_png(
        y_test,
        predictions["HOG + SVM"],
        "Confusion Matrix - HOG + SVM",
        "hog_svm_flowers_confusion_matrix.png",
    )
    plot_confusion_matrix_png(
        y_test,
        predictions["HOG + KNN"],
        "Confusion Matrix - HOG + KNN",
        "hog_knn_flowers_confusion_matrix.png",
    )

    for name, output_name in [
        ("HOG + SVM", "hog_svm_flowers_learning_curve.png"),
        ("HOG + KNN", "hog_knn_flowers_learning_curve.png"),
    ]:
        print(f"Plotting learning curve: {name}")
        sizes, train_mean, valid_mean = plot_learning_curve_png(
            models[name], f"Learning Curve - {name}", output_name, X_all, y_all
        )
        report_lines.append(f"{name} learning curve")
        for size, train_acc, cv_acc in zip(sizes, train_mean, valid_mean):
            report_lines.append(f"samples={int(size)} train={train_acc:.4f} cv={cv_acc:.4f}")
        report_lines.append("")

    plot_accuracy_comparison(metrics)
    (RESULTS_DIR / "hog_svm_knn_flowers_metrics.txt").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("Saved result files:")
    for path in sorted(RESULTS_DIR.iterdir()):
        print(f"- {path.name}")


if __name__ == "__main__":
    main()
