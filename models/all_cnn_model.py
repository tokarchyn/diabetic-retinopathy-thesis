from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

def get_all_cnn_model(input_shape, class_number, metrics, optimizer, activation='relu', kernel_reg=None, bias_reg=None):
    if kernel_reg is not None or bias_reg is not None:
        raise AttributeError('Not supported yet')

    model = Sequential()

    model.add(Conv2D(32, (7, 7), strides=1, activation=activation,  padding="valid",
                     input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (5, 5), strides=1, activation=activation,  padding="valid"))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), strides=2, activation=activation,  padding="valid"))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), strides=1, activation=activation,  padding="valid"))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), strides=2, activation=activation,  padding="valid"))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), strides=1,
                     activation=activation,  padding="valid"))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), strides=2,
                     activation=activation,  padding="valid"))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), strides=1,
                     activation=activation,  padding="valid"))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (1, 1), strides=1,
                     activation=activation,  padding="valid"))
    model.add(BatchNormalization())
    model.add(Conv2D(5, (1, 1), strides=1, activation=activation,  padding="valid"))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(class_number, activation="softmax"))

    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy", metrics=metrics)
    return model