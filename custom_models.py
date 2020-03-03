import tensorflow as tf
from tensorflow.keras import layers, Model, Input, optimizers
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # define your layers here
    def call(self, inputs, training):
        # define your model's forward pass
        # input data gets passed to this call function
        # optional training argument -  different behavior in training and inference
        pass


def model_1(input_shape, n_class):

    img_inputs = Input(shape=input_shape, name='img_inputs')

    layer = layers.Conv2D(filters = 32, kernel_size=(3, 3), padding='same', activation='relu')(img_inputs)
    layer = layers.MaxPooling2D(pool_size=(2, 2))(layer)

    layer = layers.Conv2D(filters = 64, kernel_size=(3, 3), padding='same', activation='relu')(layer)
    layer = layers.MaxPooling2D(pool_size=(2, 2))(layer)
    layer = layers.Dropout(0.25)(layer)

    layer = layers.Flatten()(layer)

    fc = layers.Dense(32, activation='relu')(layer)
    fc = layers.Dropout(0.25)(fc)
    fc = layers.Dense(n_class, activation='softmax')(fc)

    model = Model(inputs=img_inputs, outputs = fc)

    return model


def model_compile(model, lr):
    optimizer = optimizers.Adam(lr)
    model.compile(optimizer=optimizer,
                 loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model
