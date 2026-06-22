import os
import time

from hub import ClassificationHub


def main():
    dataset_dir = "flower-training"
    sample_image = os.path.join(dataset_dir, "daisy", "10140303196_b88d3d6cec.jpg")

    if not os.path.exists(sample_image):
        print(f"Sample image not found: {sample_image}")
        return

    hub = ClassificationHub(models_dir=".")
    hub.set_extractor("HOG")

    for classifier_name in ["Softmax", "SVM", "KNN"]:
        print("==================================================")
        print(f"RUNNING HOG + {classifier_name}")
        print("==================================================")

        hub.set_classifier(classifier_name)
        vocab_path = hub._get_vocab_path()
        clf_path = hub._get_classifier_path()

        if not os.path.exists(vocab_path) or not os.path.exists(clf_path):
            print(f"Model not found for HOG + {classifier_name}. Training now...")
            start_time = time.time()
            hub.train_pipeline(dataset_dir)
            print(f"Training completed in {time.time() - start_time:.2f} seconds.")

        start_time = time.time()
        result = hub.predict(sample_image)
        duration = time.time() - start_time

        print(f"Predicted class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Inference time: {duration * 1000:.2f} ms\n")


if __name__ == "__main__":
    main()
