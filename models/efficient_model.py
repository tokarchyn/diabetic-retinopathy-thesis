import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model

# Get EfficientNetB6 model, is available only from tensorflow version > 2.2
def get_efficient(train_ds, train_steps, class_number, weights, freeze_layers_number, input_shape, metrics, optimizer, activation='relu', kernel_reg=None, bias_reg=None):
    base_model = tf.keras.applications.EfficientNetB6(
        include_top=False, weights='imagenet', input_shape=input_shape)
        
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation=activation,
              kernel_regularizer=kernel_reg,
              bias_regularizer=bias_reg)(x)
    x = Dropout(0.5)(x)
    predictions = Dense(class_number, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=optimizer, loss='categorical_crossentropy')
    model.fit(train_ds, steps_per_epoch=train_steps, epochs=3, class_weight=weights)

    for layer in model.layers[:freeze_layers_number]:
        layer.trainable = False
    for layer in model.layers[freeze_layers_number:]:
        layer.trainable = True

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=metrics)
    return model