from flask import Flask, request
from flask_restful import Resource, Api, reqparse
import json

app = Flask(__name__)
api = Api(app)

# Import the required functions and constants
from model_buiding import create_model
from train_and_plot import train_model_plotting
from load_model_confusion_matrix import load_model_and_confusion_matrix
from list_models import models
from preprocess import preprocess_data

models_index = {
    0: 'efficientnet',
    1: 'mobilenet',
    2: 'inception',
    3: 'exception',
    4: 'resnet50'
}

class TrainModel(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('model_number', type=int, required=True)
        parser.add_argument('train_dir', type=str, required=True)
        parser.add_argument('val_dir', type=str, required=True)
        args = parser.parse_args()

        model_number = args['model_number']
        train_directory = args['train_dir']
        val_directory = args['val_dir']

        selected_model = models[models_index[model_number]]
        result = end_to_end_model(train_directory, val_directory, selected_model)
        return result

def end_to_end_model(train_dir, val_dir, selected_model, BATCH_SIZE=32, learning_rate=0.0001, n_epochs=10):
    train_dataset, validation_dataset, class_names = preprocess_data(train_dir, selected_model, BATCH_SIZE)
    model = create_model(train_dataset, selected_model, class_names)
    model, json_response = train_model_plotting(train_dataset, validation_dataset, learning_rate, n_epochs, model)
    model_saved_path = r'models/model_1.h5'
    model.save(model_saved_path)
    result = load_model_and_confusion_matrix(model_saved_path, val_dir, model, selected_model)

    return {
        'model_result': result,
        'training_data': json.loads(json_response),  # Parse the JSON response string
    }
api.add_resource(TrainModel, '/train_model')

if __name__ == '__main__':
    app.run(debug=True)
