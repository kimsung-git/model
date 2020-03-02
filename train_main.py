import models
import keras_utils

input_shape = (1,1,1)
model = models.model_1(input_shape)

lr = 0.001
model = models.model_compile(model, lr)

callbacks = keras_utils.get_callbacks()


class_weight = {0:1, 1:0.5}

model.fit(X_train, 
         y_train, 
         batch_size = ,
         epoch = ,
         class_weight = class_weight
         )