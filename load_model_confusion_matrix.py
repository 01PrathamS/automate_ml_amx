
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from list_models import models

def load_model_and_confusion_matrix(model_path, test_dataset, model, selected_model):
    # Load the model
    model = load_model(model_path)
    # Get class names and create mappings
    classes = sorted(os.listdir(test_dataset))
    class_names = {classes[i]: i for i in range(len(classes))}
    class_names_index = {i: classes[i] for i in range(len(classes))}

    def read_images_and_labels(data_dir):
        data, labels = [], []
        for class_name in sorted(os.listdir(data_dir)):
            label = class_names[class_name]
            for image in os.listdir(os.path.join(data_dir, class_name)):
                image_path = os.path.join(data_dir, class_name, image)
                img = cv2.imread(image_path)
                if img is not None:
                    # Resize and preprocess the image
                    img = cv2.resize(img, selected_model['IMAGE_SIZE'])
                    img = img.astype(np.float32)
                    data.append(img)
                    labels.append(label)
        return np.array(data), np.array(labels)

    validation_data, validation_labels = read_images_and_labels(test_dataset)
    test_labels = []
    for img in validation_data:
        img = np.expand_dims(img, axis=0)
        predictions = model.predict(img)
        predicted_label = class_names_index[np.argmax(predictions)]
        test_labels.append(class_names[predicted_label])

    conf_matrix = confusion_matrix(validation_labels, test_labels)
    conf_matrix_df = pd.DataFrame(conf_matrix, columns=classes, index=classes)

    class_wise_accuracy = np.diag(conf_matrix) / conf_matrix.sum(axis=1)
    class_wise_accuracy_json = {
        class_names_index[class_idx]: accuracy.item() for class_idx, accuracy in enumerate(class_wise_accuracy)
    }

    # Create a custom JSON structure for the confusion matrix
    custom_confusion_matrix = {
        "labels": [f"{classes[i]}" for i in range(len(classes))],
        "matrix": conf_matrix.tolist()
    }

    return {
        "confusion_matrix": custom_confusion_matrix,
        "class_wise_accuracy": class_wise_accuracy_json,
        "message": "Model trained successfully!"
    }