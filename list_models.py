import tensorflow as tf 


#### List of Models 

models = {
    'efficientnet': {
        'model': tf.keras.applications.EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights='imagenet'),
        'preprocess_input': tf.keras.applications.efficientnet.preprocess_input,
        'IMAGE_SIZE': (224,224)
    },
    'mobilenet': {
        'model': tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet'),
        'preprocess_input': tf.keras.applications.mobilenet_v2.preprocess_input,
        'IMAGE_SIZE': (224,224)
    },
    'inception': {
        'model': tf.keras.applications.InceptionV3(input_shape=(299, 299, 3), include_top=False, weights='imagenet'),
        'preprocess_input': tf.keras.applications.inception_v3.preprocess_input,
        'IMAGE_SIZE': (299, 299)
    },
    'exception': {
        'model': tf.keras.applications.Xception(input_shape=(299, 299, 3), include_top=False, weights='imagenet'),
        'preprocess_input': tf.keras.applications.xception.preprocess_input,
        'IMAGE_SIZE': (299, 299)
    },
    'resnet50': {
        'model': tf.keras.applications.ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet'),
        'preprocess_input': tf.keras.applications.resnet50.preprocess_input,
        'IMAGE_SIZE': (224,224)
    }
}
