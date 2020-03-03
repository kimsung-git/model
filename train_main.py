import custom_models
import keras_utils
import tensorflow as tf
import numpy as np 
from keras import backend as K
import os
from evaluate import ModelEvaluation
from tensorflow.keras import models
import datetime


def get_final_train_testset():
    mnist = tf.keras.datasets.mnist

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = np.expand_dims(x_train, axis = -1)
    x_test = np.expand_dims(x_test, axis = -1)

    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    return x_train, x_test, y_train, y_test

def train_model(model, x_train, x_test, y_train, y_test, callbacks, **kwargs):
    
    # class_weight = {0:1, 1:0.5}
    history = model.fit(x = x_train, 
        y = y_train, 
        validation_data = (x_test, y_test),
        callbacks=callbacks,
        **kwargs
        )

    return model, history
    
def get_callbacks(model_file_dir, tensorboard_dir = None):
    
    callbacks = callbacks = keras_utils.get_callbacks(model_file_dir, tensorboard_dir)
    return callbacks

def build_model(input_shape, n_class, lr):
    
    model = custom_models.model_1(input_shape, n_class)
    model = custom_models.model_compile(model, lr)
    print(model.summary())

    return model

def retrain_model(x_test, y_train, y_test, callbacks, model = None, model_file_dir = None, **kwargs):
    
    assert (model is None and model_file_dir is not None) or (model is not None and model_file_dir is None)

    if model_file_dir is not None:
        model = models.load_model(model_file_dir)
    
    model, history = train_model(model, x_train, x_test, y_train, y_test, callbacks, **kwargs)
    return model, history

if __name__ == '__main__':

    # get dataset ready
    x_train, x_test, y_train, y_test = get_final_train_testset()

    # set parameters
    input_shape = (x_train.shape)[1:]
    n_class = 10
    lr = 0.001
    tensorboard_dir = "tensorboard/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_params = {'epochs' : 10, 'batch_size' : 32}   # params passed to fit function 
    model_file_dir = os.path.join('model_dir', 'cnn.pb')  # set it to pb file!!!!! SavedModel format for god's sake!
    # tensorboard_dir = os.path.join('tensorboard', 'test')
    
    # train 
    callbacks = get_callbacks(model_file_dir, tensorboard_dir)
    model = build_model(input_shape, n_class, lr)
    model, history = train_model(model, x_train, x_test, y_train, y_test, callbacks, **model_params)

    # # retrain 
    model, history = retrain_model(x_test, y_train, y_test, callbacks, model = None, model_file_dir = model_file_dir, **model_params)

    # predict and evaluate
    # model = models.load_model(model_file_dir)   # load model
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis = 1)

    evals = ModelEvaluation(y_test, y_pred) 
    report = evals.classification_report()
    cm = evals.confusion_matrix()

    print(report)
    print(cm)
