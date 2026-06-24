#!/usr/bin/env python3
"""Train all missing classifier+extractor combinations, reusing existing vocabs."""

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

from preprocess import extract_flower_mask, preprocess_single_image
from services import (
    HOGService, ORBService, SIFTService, VisualVocabulary,
    KNNService, RandomForestService, SVMService, XGBoostService,
    extract_hsv_histogram,
)

DATASET_DIR = "flower-training"
MODELS_DIR = "."
CLASSES = ["bellflower", "daisy", "dandelion", "lotus", "rose", "sunflower", "tulip"]

COMBOS = [
    ("SIFT", "RandomForest"),
    ("SIFT", "XGBoost"),
    ("SIFT", "SVM"),
    ("SIFT", "KNN"),
    ("ORB", "RandomForest"),
    ("ORB", "XGBoost"),
    ("ORB", "SVM"),
    ("ORB", "KNN"),
    ("HOG", "RandomForest"),
    ("HOG", "XGBoost"),
    ("HOG", "SVM"),
    ("HOG", "KNN"),
]


def get_extractor(name):
    if name == "SIFT":
        return SIFTService()
    if name == "ORB":
        return ORBService()
    if name == "HOG":
        return HOGService()
    raise ValueError(name)


def get_classifier(name):
    if name == "RandomForest":
        return RandomForestService()
    if name == "XGBoost":
        return XGBoostService()
    if name == "SVM":
        return SVMService()
    if name == "KNN":
        return KNNService()
    raise ValueError(name)


def vocab_path(ext):
    return os.path.join(MODELS_DIR, f"vocab_{ext.lower()}_opt.joblib")


def clf_path(ext, clf):
    return os.path.join(MODELS_DIR, f"model_{clf.lower()}_{ext.lower()}_opt.joblib")


def load_image_paths():
    paths, labels = [], []
    for idx, cls in enumerate(CLASSES):
        cls_dir = os.path.join(DATASET_DIR, cls)
        if not os.path.exists(cls_dir):
            print(f"  Missing: {cls_dir}")
            continue
        for name in os.listdir(cls_dir):
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                paths.append(os.path.join(cls_dir, name))
                labels.append(idx)
    return paths, labels


def extract_all(image_paths, labels, extractor):
    descs_list, color_list, valid_labels = [], [], []
    for i, img_path in enumerate(image_paths):
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
            cleaned = preprocess_single_image(img, targetedSize=(256, 256))
            mask = extract_flower_mask(cleaned)
            descs = extractor.extract_features(cleaned, mask=mask)
            color = extract_hsv_histogram(cleaned, mask=mask)
            descs_list.append(descs)
            color_list.append(color)
            valid_labels.append(labels[i])
        except Exception as e:
            print(f"  Skip {img_path}: {e}")
        if (i + 1) % 500 == 0 or (i + 1) == len(image_paths):
            print(f"  {i+1}/{len(image_paths)} processed")
    return descs_list, color_list, np.array(valid_labels)


def fuse(descs, color, vocab):
    hist = vocab.transform(descs)
    h = hist / (np.linalg.norm(hist, ord=2) + 1e-6)
    c = color / (np.linalg.norm(color, ord=2) + 1e-6)
    fv = np.hstack([h, 0.5 * c])
    fv /= np.linalg.norm(fv, ord=2) + 1e-6
    return fv


def main():
    image_paths, labels = load_image_paths()
    print(f"Dataset: {len(image_paths)} images\n")

    for ext_name in ["SIFT", "ORB", "HOG"]:
        missing = [c for e, c in COMBOS if e == ext_name and not os.path.exists(clf_path(ext_name, c))]
        if not missing:
            print(f"[{ext_name}] All models present, skipping.")
            continue

        print(f"\n{'='*50}")
        print(f"[{ext_name}] Cần train: {missing}")
        print(f"{'='*50}")

        extractor = get_extractor(ext_name)
        print(f"Extracting features...")
        descs_list, color_list, y = extract_all(image_paths, labels, extractor)

        vpath = vocab_path(ext_name)
        vocab = VisualVocabulary(n_clusters=500)
        if os.path.exists(vpath):
            print(f"  Load vocab: {vpath}")
            vocab.load(vpath)
        else:
            print(f"  Training vocab...")
            train_idx, _ = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)
            vocab.fit([descs_list[i] for i in train_idx])
            vocab.save(vpath)
            print(f"  Saved vocab: {vpath}")

        indices = np.arange(len(y))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=y, random_state=42)
        X_train = np.array([fuse(descs_list[i], color_list[i], vocab) for i in train_idx])
        X_test = np.array([fuse(descs_list[i], color_list[i], vocab) for i in test_idx])
        y_train, y_test = y[train_idx], y[test_idx]

        for clf_name in missing:
            cpath = clf_path(ext_name, clf_name)
            print(f"\n  [Train] {ext_name}+{clf_name}...")
            clf = get_classifier(clf_name)
            clf.fit(X_train, y_train, classes=CLASSES)
            clf.save(cpath)

            train_preds = np.array([CLASSES.index(clf.predict(x)) for x in X_train])
            test_preds = np.array([CLASSES.index(clf.predict(x)) for x in X_test])
            print(f"  Train: {np.mean(train_preds == y_train)*100:.2f}%  Test: {np.mean(test_preds == y_test)*100:.2f}%")
            print(f"  Saved: {cpath}")

    print("\nXong!")


if __name__ == "__main__":
    main()
