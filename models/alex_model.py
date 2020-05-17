from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

def get_alex_model(input_shape, class_number, metrics, lr):
    model = Sequential()

    model.add(Conv2D(96, (11, 11), strides=4, activation="relu",  padding="same",
                     input_shape=input_shape))
    model.add(MaxPooling2D((2, 2), strides=2))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (11, 11), strides=1,
                     padding="same", activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=2))
    model.add(BatchNormalization())

    model.add(Conv2D(384, (3, 3), strides=1,
                     padding="same", activation="relu"))
    model.add(Conv2D(384, (3, 3), strides=1,
                     padding="same", activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=2))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (3, 3), strides=1,
                     padding="same", activation="relu"))
    model.add(Conv2D(256, (3, 3),  padding="same", activation="relu"))
    model.add(Conv2D(256, (3, 3),  padding="same", activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=2))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (3, 3), strides=1,
                     padding="same",  activation="relu"))
    model.add(Conv2D(256, (3, 3),  padding="same", activation="relu"))
    model.add(Conv2D(256, (3, 3),  padding="same", activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=2))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(class_number, activation="softmax"))

    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy", metrics=metrics)
    return model