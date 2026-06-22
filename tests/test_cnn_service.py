import numpy as np
import pytest
import os
import tempfile


def make_tiny_cnn_model():
    """Tạo model CNN nhỏ để test (không cần train thật)."""
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(4, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(7, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return model


def test_cnn_service_not_fitted_initially():
    from services import CNNService
    svc = CNNService()
    assert svc.is_fitted is False


def test_cnn_service_load_sets_fitted():
    from services import CNNService
    svc = CNNService()
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        tmp_path = f.name
    try:
        model = make_tiny_cnn_model()
        model.save(tmp_path)
        classes = ["bellflower", "daisy", "dandelion", "lotus", "rose", "sunflower", "tulip"]
        svc.load(tmp_path, classes)
        assert svc.is_fitted is True
        assert svc.classes == classes
    finally:
        os.unlink(tmp_path)


def test_cnn_service_predict_returns_class_name():
    from services import CNNService
    svc = CNNService()
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        tmp_path = f.name
    try:
        model = make_tiny_cnn_model()
        model.save(tmp_path)
        classes = ["bellflower", "daisy", "dandelion", "lotus", "rose", "sunflower", "tulip"]
        svc.load(tmp_path, classes)
        # Tạo ảnh giả BGR 100x100
        fake_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = svc.predict(fake_img)
        assert result in classes
    finally:
        os.unlink(tmp_path)


def test_cnn_service_predict_proba_shape():
    from services import CNNService
    svc = CNNService()
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        tmp_path = f.name
    try:
        model = make_tiny_cnn_model()
        model.save(tmp_path)
        classes = ["bellflower", "daisy", "dandelion", "lotus", "rose", "sunflower", "tulip"]
        svc.load(tmp_path, classes)
        fake_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        probs = svc.predict_proba(fake_img)
        assert probs.shape == (7,)
        assert abs(probs.sum() - 1.0) < 1e-5
    finally:
        os.unlink(tmp_path)


def test_cnn_service_predict_raises_if_not_loaded():
    from services import CNNService
    svc = CNNService()
    fake_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="chưa được tải"):
        svc.predict(fake_img)


def test_cnn_service_predict_proba_raises_if_not_loaded():
    from services import CNNService
    svc = CNNService()
    fake_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="chưa được tải"):
        svc.predict_proba(fake_img)
