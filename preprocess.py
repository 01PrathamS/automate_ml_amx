import tensorflow as tf 
from list_models import models

## train_dir
## selected_model
## batch_size


def preprocess_data(train_dir,selected_model,BATCH_SIZE):
    train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir, 
                                                            shuffle=True, 
                                                            batch_size=BATCH_SIZE,
                                                            image_size=selected_model['IMAGE_SIZE'], 
                                                            label_mode='categorical')
    class_names = train_dataset.class_names
    ## train_test_split_ratio : 0.2
    train_batches = tf.data.experimental.cardinality(train_dataset) 
    validation_dataset = train_dataset.take(train_batches // 5) 
    train_dataset = train_dataset.skip(train_batches // 5) 
    print('Step 1: PreProcessing Completed')
    print(len(train_dataset), len(validation_dataset))
    # print(len(train_dataset)*32, len(validation_dataset)*32)
    return train_dataset, validation_dataset, class_names


train_directory = r"D:\crop_0\crop_data\test"
selected_model = models['efficientnet']
BATCH_SIZE=32
train_dataset, validation_dataset, class_names = preprocess_data(train_directory, selected_model, BATCH_SIZE)

import matplotlib.pyplot as plt

for images, labels in train_dataset.take(1):  # Change 1 to the number of batches you want to visualize
    # Loop through the images in the batch
    for i in range(len(images)):
        # Display the image and its corresponding label
        plt.figure(figsize=(4, 4))
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(f"Label: {labels[i].numpy()}")
        plt.axis("off")
        plt.show()