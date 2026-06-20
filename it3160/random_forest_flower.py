"""
Random Forest Flower Image Classification
Built following the same pipeline as the Softmax notebook:
load images -> resize 32x32 -> train/val/test split -> flatten -> normalize -> train Random Forest -> tune hyperparameters -> evaluate -> save model.
"""

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

CLASSES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
IMG_SIZE = (32, 32)
DATASET_DIR = 'flower-training'


def load_flower_data(data_dir=DATASET_DIR, img_size=IMG_SIZE):
    """Load flower images from folders and convert them to RGB arrays."""
    X = []
    y = []

    for label, cls in enumerate(CLASSES):
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.exists(cls_dir):
            print(f"Warning: folder not found: {cls_dir}")
            continue

        for img_name in os.listdir(cls_dir):
            img_path = os.path.join(cls_dir, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(img_size)
                X.append(np.array(img))
                y.append(label)
            except Exception:
                pass

    X = np.array(X).astype('float64')
    y = np.array(y)

    np.random.seed(42)
    idxs = np.random.permutation(len(X))
    return X[idxs], y[idxs]


def add_basic_color_features(X_images):
    """
    Optional improvement for Random Forest:
    combine raw pixels with simple color statistics.
    X_images shape: (N, 32, 32, 3)
    """
    X_flat = X_images.reshape(X_images.shape[0], -1)
    mean_rgb = np.mean(X_images, axis=(1, 2))
    std_rgb = np.std(X_images, axis=(1, 2))
    min_rgb = np.min(X_images, axis=(1, 2))
    max_rgb = np.max(X_images, axis=(1, 2))
    return np.hstack([X_flat, mean_rgb, std_rgb, min_rgb, max_rgb])


def predict_single_image(model, image_path, mean_image=None, use_extra_features=False):
    """Predict one image using the trained Random Forest model."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype('float64')

    if use_extra_features:
        X = add_basic_color_features(arr.reshape(1, 32, 32, 3))
    else:
        X = arr.reshape(1, -1)
        if mean_image is not None:
            X = X - mean_image

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    return CLASSES[pred], proba


def main():
    print('Loading flower dataset...')
    X, y = load_flower_data()
    print('Full dataset:', X.shape, y.shape)

    # Same idea as the Softmax notebook: first split train/test.
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Then split train into train/validation.
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_raw, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    print('Train:', X_train_raw.shape)
    print('Validation:', X_val_raw.shape)
    print('Test:', X_test_raw.shape)

    # Choose one feature style.
    # Option A: use raw pixels, like the Softmax notebook.
    use_extra_features = False

    if use_extra_features:
        X_train = add_basic_color_features(X_train_raw)
        X_val = add_basic_color_features(X_val_raw)
        X_test = add_basic_color_features(X_test_raw)
        mean_image = None
    else:
        X_train = X_train_raw.reshape(X_train_raw.shape[0], -1)
        X_val = X_val_raw.reshape(X_val_raw.shape[0], -1)
        X_test = X_test_raw.reshape(X_test_raw.shape[0], -1)

        # Keep the same normalization idea as Softmax.
        mean_image = np.mean(X_train, axis=0)
        X_train = X_train - mean_image
        X_val = X_val - mean_image
        X_test = X_test - mean_image

    print('Feature shape:', X_train.shape)

    # Hyperparameter tuning for Random Forest.
    results = {}
    best_val = -1
    best_rf = None

    n_estimators_list = [100, 200, 300]
    max_depth_list = [None, 10, 20, 30]
    min_samples_leaf_list = [1, 2, 4]

    for n_estimators in n_estimators_list:
        for max_depth in max_depth_list:
            for min_samples_leaf in min_samples_leaf_list:
                rf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'
                )
                rf.fit(X_train, y_train)

                train_acc = accuracy_score(y_train, rf.predict(X_train))
                val_acc = accuracy_score(y_val, rf.predict(X_val))

                params = (n_estimators, max_depth, min_samples_leaf)
                results[params] = (train_acc, val_acc)

                print(f'n_estimators={n_estimators}, max_depth={max_depth}, '
                      f'min_samples_leaf={min_samples_leaf}: train={train_acc:.4f}, val={val_acc:.4f}')

                if val_acc > best_val:
                    best_val = val_acc
                    best_rf = rf

    print('\nBest validation accuracy:', best_val)
    print('Best model:', best_rf)

    # Test evaluation.
    y_test_pred = best_rf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print('\nRandom Forest test accuracy:', test_accuracy)
    print('\nClassification report:')
    print(classification_report(y_test, y_test_pred, target_names=CLASSES))
    print('\nConfusion matrix:')
    print(confusion_matrix(y_test, y_test_pred))

    # Save model and preprocessing data.
    joblib.dump({
        'model': best_rf,
        'classes': CLASSES,
        'img_size': IMG_SIZE,
        'mean_image': mean_image,
        'use_extra_features': use_extra_features
    }, 'best_random_forest_flower.joblib')

    print('\nSaved: best_random_forest_flower.joblib')


if __name__ == '__main__':
    main()
