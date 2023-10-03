from model_buiding import create_model 
from train_and_plot import train_model_plotting 
from load_model_confusion_matrix import load_model_and_confusion_matrix 
from list_models import models 
from preprocess import preprocess_data

models_index = {0: 'efficientnet',  #'efficientnet'
                1: 'mobilenet',  # 'mobilenet'
                2: 'inception',  #'inception'
                3: 'exception',  # 'exception'
                4: 'resnet50'}    # 'resnet50'


def end_to_end_model(train_dir,val_dir, selected_model=models['resnet50'],BATCH_SIZE = 32,learning_rate=0.0001,n_epochs=1):

    train_dataset, validation_dataset, class_names = preprocess_data(train_dir,selected_model,BATCH_SIZE)

    model = create_model(train_dataset, selected_model, class_names)

    model = train_model_plotting(train_dataset, validation_dataset, learning_rate, n_epochs, model)

    model_saved_path = r'models/model_1.h5'
    model.save(model_saved_path)
    load_model_and_confusion_matrix(model_saved_path, val_dir,model, selected_model)

    return 

# model_number = int(input("Enter a number on which model you want to train classifier: "))

# train_directory = r"D:\crop_0\crop_data\test"
# val_directory = r"D:\crop_0\crop_data\test"
# end_to_end_model(train_directory, val_directory,models[models_index[model_number]])
