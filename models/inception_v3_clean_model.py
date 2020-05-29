from tensorflow.keras.optimizers import *
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model

def get_inception_v3_clean(class_number, input_shape, metrics, optimizer, activation='relu', kernel_reg=None, bias_reg=None):
    base_model = InceptionV3(weights=None,
                             include_top=False,
                             input_shape=input_shape)
    x = base_model.output
    # x = Dropout(0.5)(x)
    x = Flatten()(x)
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
    
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=metrics)
    return model