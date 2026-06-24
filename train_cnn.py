import os

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

DATASET_DIR = "flowers"
MODEL_OUTPUT = "cnn_flower_model.h5"
IMG_SIZE = (128, 128)
CLASSES = ["bellflower", "daisy", "dandelion", "lotus", "rose", "sunflower", "tulip"]
BATCH_SIZE = 32
EPOCHS = 80
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE


def resolve_dataset_dir(dataset_dir):
    candidates = [
        dataset_dir,
        os.path.join(dataset_dir, "flowers"),
        "flower-training",
        os.path.join("it3160", dataset_dir),
    ]
    for candidate in candidates:
        if all(os.path.isdir(os.path.join(candidate, cls)) for cls in CLASSES):
            return candidate

    raise FileNotFoundError(
        "Cannot find a dataset folder containing all class subfolders. Tried: "
        + ", ".join(candidates)
    )


def load_dataset(dataset_dir):
    dataset_dir = resolve_dataset_dir(dataset_dir)
    X, y = [], []

    for label_idx, cls in enumerate(CLASSES):
        cls_dir = os.path.join(dataset_dir, cls)
        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                continue

            img_path = os.path.join(cls_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Skipping unreadable image: {img_path}")
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
            X.append(img.astype(np.float32) / 255.0)
            y.append(label_idx)

    if not X:
        raise ValueError(f"No training images were loaded from: {dataset_dir}")

    return np.array(X), np.array(y), dataset_dir


def make_dataset(X, y, training=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(len(X), seed=SEED, reshuffle_each_iteration=True)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


def build_model(num_classes=7):
    """Build the augmented, regularized CNN used for training."""
    regularizer = tf.keras.regularizers.l2(1e-4)

    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(IMG_SIZE[1], IMG_SIZE[0], 3)),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomContrast(0.15),
            tf.keras.layers.Conv2D(
                32, (3, 3), padding="same", use_bias=False, kernel_regularizer=regularizer
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Conv2D(
                64, (3, 3), padding="same", use_bias=False, kernel_regularizer=regularizer
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Conv2D(
                128, (3, 3), padding="same", use_bias=False, kernel_regularizer=regularizer
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(
                192, (3, 3), padding="same", use_bias=False, kernel_regularizer=regularizer
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=regularizer),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    tf.keras.utils.set_random_seed(SEED)

    print("Loading dataset...")
    X, y, dataset_root = load_dataset(DATASET_DIR)
    print(f"Dataset: {dataset_root}")
    print(f"Total images: {len(X)}, shape: {X.shape}")
    for class_idx, cls in enumerate(CLASSES):
        print(f"- {cls}: {np.sum(y == class_idx)} images")

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=SEED
    )
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    train_ds = make_dataset(X_train, y_train, training=True)
    val_ds = make_dataset(X_val, y_val)
    test_ds = make_dataset(X_test, y_test)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(CLASSES)),
        y=y_train,
    )
    class_weight = {idx: float(weight) for idx, weight in enumerate(class_weights)}

    model = build_model(num_classes=len(CLASSES))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=12, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_OUTPUT, monitor="val_accuracy", mode="max", save_best_only=True, verbose=1
        ),
    ]

    print("\nTraining CNN...")
    model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    train_loss, train_acc = model.evaluate(make_dataset(X_train, y_train), verbose=0)
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)

    print("\n=== RESULTS ===")
    print(f"Train accuracy: {train_acc * 100:.2f}%")
    print(f"Val accuracy: {val_acc * 100:.2f}%")
    print(f"Test accuracy: {test_acc * 100:.2f}%")
    print(f"Saved model: {MODEL_OUTPUT}")
