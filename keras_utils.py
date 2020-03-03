from tensorflow.keras import callbacks
import model_config as c

def get_callbacks(model_file_dir, tensorboard_dir = None):

    # filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    model_chcekpoint = callbacks.ModelCheckpoint(model_file_dir, monitor='val_acc', verbose=1, 
                                                save_best_only=True, mode='auto')

    earlystop = callbacks.EarlyStopping(monitor = 'val_loss',
                                        patience = 7,
                                        verbose = 1,
                                        mode = 'auto')


    # initial_learning_rate = 0.1
    # lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate,
    #     decay_step = 100000,
    #     decay_rate = 0.96,
    #     staircase = True
    # )

    if tensorboard_dir is not None:
        tensorboard = callbacks.TensorBoard(log_dir = tensorboard_dir)
    

    return [model_chcekpoint, tensorboard, earlystop]


def get_loss():
    pass
