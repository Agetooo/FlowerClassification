import os
import gc
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from preprocess import preprocess_single_image, extract_flower_mask
from services import (
    SIFTService,
    ORBService,
    RandomForestService,
    XGBoostService,
    extract_hsv_histogram,
    VisualVocabulary,
)

DATASET_DIR = "flower-training"
CLASSES = ["bellflower", "daisy", "dandelion", "lotus", "rose", "sunflower", "tulip"]

def safe_load_image(path):
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None

def train_extractor_models(extractor_name):
    print(f"\n==================================================")
    print(f"Training pipeline for Extractor: {extractor_name}")
    print(f"==================================================")
    
    # 1. Load image paths
    image_paths = []
    labels = []
    for label_idx, cls in enumerate(CLASSES):
        cls_dir = os.path.join(DATASET_DIR, cls)
        if not os.path.exists(cls_dir):
            continue
        for filename in os.listdir(cls_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(cls_dir, filename))
                labels.append(label_idx)
                
    print(f"Loaded {len(image_paths)} images.")
    
    # Split: Train (80%) and Test (20%)
    X_train_paths, X_test_paths, y_train, y_test = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    # Further split Train into Train (90% of Train) and Validation (10% of Train)
    X_tr_paths, X_val_paths, y_tr, y_val = train_test_split(
        X_train_paths, y_train, test_size=0.1, stratify=y_train, random_state=42
    )
    
    if extractor_name == "SIFT":
        extractor = SIFTService()
    else:
        extractor = ORBService()
        
    # Fit visual vocabulary on 1000 sampled train images
    print("Sampling training images for vocabulary fitting...")
    np.random.seed(42)
    fit_paths = np.random.choice(X_tr_paths, min(len(X_tr_paths), 1000), replace=False)
    
    fit_descs = []
    for idx, path in enumerate(fit_paths):
        img = safe_load_image(path)
        if img is None:
            continue
        cleaned = preprocess_single_image(img)
        mask = extract_flower_mask(cleaned)
        descs = extractor.extract_features(cleaned, mask=mask)
        if len(descs) > 0:
            s_idx = np.random.choice(len(descs), min(len(descs), 100), replace=False)
            fit_descs.append(descs[s_idx])
        if (idx + 1) % 100 == 0:
            gc.collect()
            
    flat_fit_descs = np.vstack(fit_descs)
    print(f"Fitting K-Means vocabulary with 500 clusters...")
    vocab = VisualVocabulary(n_clusters=500)
    vocab.kmeans.fit(flat_fit_descs)
    vocab.is_fitted = True
    
    # Save vocabulary
    vocab_path = f"vocab_{extractor_name.lower()}_opt.joblib"
    vocab.save(vocab_path)
    print(f"Saved vocabulary to {vocab_path}")
    
    del fit_descs, flat_fit_descs
    gc.collect()
    
    # Extract features for all splits
    def get_fused_features(paths):
        features = []
        for idx, path in enumerate(paths):
            img = safe_load_image(path)
            if img is None:
                features.append(np.zeros(628, dtype=np.float32))
                continue
            cleaned = preprocess_single_image(img)
            mask = extract_flower_mask(cleaned)
            descs = extractor.extract_features(cleaned, mask=mask)
            color = extract_hsv_histogram(cleaned, mask=mask)
            
            # Project to vocabulary
            hist = vocab.transform(descs)
            hist_l2 = hist / (np.linalg.norm(hist, ord=2) + 1e-6)
            color_l2 = color / (np.linalg.norm(color, ord=2) + 1e-6)
            fused = np.hstack([hist_l2, 0.5 * color_l2])
            fused /= np.linalg.norm(fused, ord=2) + 1e-6
            features.append(fused)
            
            if (idx + 1) % 500 == 0:
                gc.collect()
        return np.array(features)
        
    print("Extracting features for Train split...")
    X_train_fused = get_fused_features(X_tr_paths)
    print("Extracting features for Test split...")
    X_test_fused = get_fused_features(X_test_paths)
    
    y_tr_arr = np.array(y_tr)
    y_test_arr = np.array(y_test)
    
    # Train RandomForest
    print("Training RandomForest classifier...")
    rf = RandomForestService()
    rf.fit(X_train_fused, y_tr_arr, classes=CLASSES)
    rf_path = f"model_randomforest_{extractor_name.lower()}_opt.joblib"
    rf.save(rf_path)
    print(f"Saved RandomForest model to {rf_path}")
    
    # Train XGBoost
    print("Training XGBoost classifier...")
    xgb = XGBoostService()
    xgb.fit(X_train_fused, y_tr_arr, classes=CLASSES)
    xgb_path = f"model_xgboost_{extractor_name.lower()}_opt.joblib"
    xgb.save(xgb_path)
    print(f"Saved XGBoost model to {xgb_path}")
    
    # Evaluate
    rf_preds = np.array([CLASSES.index(rf.predict(x)) for x in X_test_fused])
    xgb_preds = np.array([CLASSES.index(xgb.predict(x)) for x in X_test_fused])
    print(f"RandomForest Test Accuracy: {np.mean(rf_preds == y_test_arr)*100:.2f}%")
    print(f"XGBoost Test Accuracy: {np.mean(xgb_preds == y_test_arr)*100:.2f}%")
    
    del X_train_fused, X_test_fused
    gc.collect()

def main():
    train_extractor_models("SIFT")
    train_extractor_models("ORB")
    print("All production models successfully trained on the full dataset and saved!")

if __name__ == "__main__":
    main()
