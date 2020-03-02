from tensorflow.keras import layers, Model, Input, optimizers


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # define your layers here
    def call(self, inputs, training):
        # define your model's forward pass
        # input data gets passed to this call function
        # optional training argument -  different behavior in training and inference
        

def model_1(input_shape):

    pass


def model_compile(model, lr):

    optimizer = optimizers.Adadelta(lr)
    model.compile(optimizer = optimizer, 
                  loss = ,
                  metrics = ,
                  )
    
    return model
