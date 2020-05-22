from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

def get_vgg_model(input_shape, class_number, metrics, optimizer, activation='relu', kernel_reg=None, bias_reg=None):
    model = Sequential()
    model.add(Conv2D(32, (7, 7), padding="same", activation=activation,
                     kernel_regularizer=kernel_reg,
                     bias_regularizer=bias_reg,
                     input_shape=input_shape))
    model.add(MaxPooling2D((3, 3)))
    model.add(BatchNormalization())

    model.add(Conv2D(32, (3, 3), padding="same", activation=activation,
                     kernel_regularizer=kernel_reg,
                     bias_regularizer=bias_reg))
    model.add(Conv2D(32, (3, 3), padding="same", activation=activation,                    
                     kernel_regularizer=kernel_reg,
                     bias_regularizer=bias_reg))
    model.add(MaxPooling2D((3, 3)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), padding="same", activation=activation,
                     kernel_regularizer=kernel_reg,
                     bias_regularizer=bias_reg))
    model.add(Conv2D(64, (3, 3), padding="same", activation=activation,
                     kernel_regularizer=kernel_reg,
                     bias_regularizer=bias_reg))
    model.add(Conv2D(64, (3, 3), padding="same", activation=activation,
                     kernel_regularizer=kernel_reg,
                     bias_regularizer=bias_reg))
    model.add(MaxPooling2D((3, 3)))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (3, 3), padding="same", activation=activation,
                     kernel_regularizer=kernel_reg,
                     bias_regularizer=bias_reg))
    model.add(Conv2D(256, (3, 3), padding="same", activation=activation,
                     kernel_regularizer=kernel_reg,
                     bias_regularizer=bias_reg))
    model.add(Conv2D(256, (3, 3), padding="same", activation=activation,
                     kernel_regularizer=kernel_reg,
                     bias_regularizer=bias_reg))
    model.add(MaxPooling2D((3, 3)))
    model.add(BatchNormalization())

    model.add(Conv2D(512, (3, 3), padding="same", activation=activation,
                     kernel_regularizer=kernel_reg,
                     bias_regularizer=bias_reg))
    model.add(Conv2D(512, (3, 3), padding="same", activation=activation,
                     kernel_regularizer=kernel_reg,
                     bias_regularizer=bias_reg))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(1024, activation=activation,
                     kernel_regularizer=kernel_reg,
                     bias_regularizer=bias_reg))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation=activation,
                     kernel_regularizer=kernel_reg,
                     bias_regularizer=bias_reg))
    model.add(Dropout(0.5))
    model.add(Dense(class_number, activation="softmax"))

    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=metrics)
    return model