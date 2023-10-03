from list_models import models 
import tensorflow as tf 
import matplotlib.pyplot as plt
import json 

## train_dataset, validation_dataset
## learning_rate 
## n_epochs
## model 

def train_model_plotting(train_dataset, validation_dataset, learning_rate, n_epochs, selected_model):
    base_learning_rate = learning_rate
    selected_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate), 
              loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
              metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')])
    
    history = selected_model.fit(train_dataset, 
                                 epochs=n_epochs, 
                                 validation_data=validation_dataset, 
                                 verbose=0) 
    # print(history.history)
    # print(history.history.keys())

    acc = history.history['accuracy']  
    val_acc = history.history['val_accuracy'] 
    loss = history.history['loss'] 
    val_loss = history.history['val_loss']  
    
    results = {
        'acc': acc,
        'val_acc': val_acc,
        'loss': loss,
        'val_loss': val_loss
    }

    json_response = json.dumps(results)

    return selected_model, json_response

    # print('Step 3: Plotting and Training Completed')
    # return selected_model








