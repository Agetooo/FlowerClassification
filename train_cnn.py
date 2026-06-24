import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

DATASET_DIR = "flower-training"
MODEL_OUTPUT = "cnn_flower_model.h5"
IMG_SIZE = (64, 64)
CLASSES = ["bellflower", "daisy", "dandelion", "lotus", "rose", "sunflower", "tulip"]
BATCH_SIZE = 32
EPOCHS = 50


def load_dataset(dataset_dir):
    X, y = [], []
    for label_idx, cls in enumerate(CLASSES):
        cls_dir = os.path.join(dataset_dir, cls)
        if not os.path.exists(cls_dir):
            print(f"Bỏ qua: {cls_dir}")
            continue
        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                continue
            img_path = os.path.join(cls_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMG_SIZE)
            X.append(img.astype(np.float32) / 255.0)
            y.append(label_idx)
    return np.array(X), np.array(y)


def build_model(num_classes=7):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(64, 64, 3)),

        # Data augmentation — chỉ active khi training=True, tắt tự động lúc predict
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),

        # Khối 1
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        # Khối 2
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        # Khối 3
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        # GlobalAveragePooling: 8×8×128 → 128
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


if __name__ == "__main__":
    print("Đang đọc dataset...")
    X, y = load_dataset(DATASET_DIR)
    print(f"Tổng số ảnh: {len(X)}, shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    model = build_model(num_classes=len(CLASSES))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_OUTPUT, monitor='val_loss', save_best_only=True, verbose=1
        )
    ]

    print("\nBắt đầu train CNN...")
    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n=== KẾT QUẢ ===")
    print(f"Test accuracy: {acc * 100:.2f}%")
    print(f"Model đã lưu tại: {MODEL_OUTPUT}")
