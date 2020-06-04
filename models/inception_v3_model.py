from tensorflow.keras.optimizers import *
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model

def get_inception_v3(train_ds, train_steps, class_number, weights, freeze_layers_number, input_shape, metrics, optimizer, activation='relu', kernel_reg=None, bias_reg=None):
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet',
                             include_top=False,
                             input_shape=input_shape)
    x = base_model.output

    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation=activation,
                    kernel_regularizer=kernel_reg,
                    bias_regularizer=bias_reg)(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation=activation,
                    kernel_regularizer=kernel_reg,
                    bias_regularizer=bias_reg)(x)
    x = Dropout(0.5)(x)
    predictions = Dense(class_number, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=optimizer, loss='categorical_crossentropy')

    # train the model on the new data for a few epoch
    model.fit(train_ds, steps_per_epoch=train_steps,
              epochs=3, class_weight=weights)

    for layer in model.layers[:freeze_layers_number]:
        layer.trainable = False
    for layer in model.layers[freeze_layers_number:]:
        layer.trainable = True

    # recompile
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=metrics)
    return model