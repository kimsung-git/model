from tensorflow.keras import callbacks
import model_config as c

def get_callbacks(model_file_dir, tensorboard_dir):


    # file format can be either h5, ckpt, or pb(not sure about .pb)
    # filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    model_chcekpoint = callbacks.ModelCheckpoint(model_file_dir, monitor='val_accuracy', verbose=1, save_best_only=True, monitor = 'val_loss', mode='max')

    # filepath="weights.best.hdf5"
    # model_chcekpoint = callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')



    # initial_learning_rate = 0.1
    # lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate,
    #     decay_step = 100000,
    #     decay_rate = 0.96,
    #     staircase = True
    # )


    tensorboard = callbacks.TensorBoard(log_dir = tensorboard_dir)


    earlystop = callbacks.EarlyStopping(monitor = 'val_loss',
                            min_delta = 0,
                            patience = 10,
                            verbose = 1,
                            restore_best_weights = True)

    return [model_chcekpoint, tensorboard, earlystop]


def get_loss():
    pass
