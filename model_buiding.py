import os 
import numpy as np 
import tensorflow as tf 
from list_models import models 

## train_dataset
## model


def create_model(train_dataset,selected_model, class_names):
    ## DATA AUGMENTATION
    data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'), 
    tf.keras.layers.RandomRotation(0.2)])

    ## CLASSES

    class_names  = class_names
    num_classes = len(np.unique(class_names))
    ## MODEL   ## Addding layers on top of Base model
    
    base_model = selected_model['model']
    image_batch, label_batch = next(iter(train_dataset)) 
    feature_batch = base_model(image_batch) 
    base_model.trainable = False 
    preprocess_input = selected_model['preprocess_input']
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D() 
    feature_batch_average = global_average_layer(feature_batch) 
    prediction_layer = tf.keras.layers.Dense(num_classes) 
    prediction_batch = prediction_layer(feature_batch_average) 
    inputs = tf.keras.Input(shape=(selected_model['IMAGE_SIZE']+(3,))) 
    x = data_augmentation(inputs) 
    x = preprocess_input(x) 
    x = base_model(x, training=False) 
    x = global_average_layer(x) 
    x = tf.keras.layers.Dropout(0.2)(x) 
    outputs = prediction_layer(x) 
    model = tf.keras.Model(inputs, outputs) 
    print(f'selected model: {selected_model["model"]}')
    print('Step 2: Model Building Completed')
    return model 