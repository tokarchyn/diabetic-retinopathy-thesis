from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

def get_vgg_model(input_shape, class_number, metrics, lr, activation='relu'):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", activation=activation,
                     input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), padding="same", activation=activation))
    model.add(MaxPooling2D((3, 3)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), padding="same", activation=activation))
    model.add(Conv2D(64, (3, 3), padding="same", activation=activation))
    model.add(Conv2D(64, (3, 3), padding="same", activation=activation))
    model.add(MaxPooling2D((3, 3)))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (3, 3), padding="same", activation=activation))
    model.add(Conv2D(256, (3, 3), padding="same", activation=activation))
    model.add(Conv2D(256, (3, 3), padding="same", activation=activation))
    model.add(MaxPooling2D((3, 3)))
    model.add(BatchNormalization())

    model.add(Conv2D(512, (3, 3), padding="same", activation=activation))
    model.add(Conv2D(512, (3, 3), padding="same", activation=activation))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(1024, activation=activation))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation=activation))
    model.add(Dropout(0.5))
    model.add(Dense(class_number, activation="softmax"))

    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=metrics)
    return model