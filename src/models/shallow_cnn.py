"""
Function for building shallow CNN.
"""
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input
from keras.models import Model


def build_shallow_cnn(config: dict = None) -> Model:
    """
    Function allowing to build Shallow CNN using Functional API.
    """
    if config is None:
        config = {}

    input_shape = config.get('input_shape', (8, 8, 12))

    inputs = Input(shape=input_shape, name='input_layer')

    # Blok 1: 32 filters
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_1')(inputs)
    x = Dropout(0.2, name='dropout_1')(x)

    # Blok 2: 64 filters
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_2')(x)
    x = Dropout(0.2, name='dropout_2')(x)

    # Blok 3: 64 filters
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_3')(x)
    x = Dropout(0.2, name='dropout_3')(x)

    # Blok 4: 128 filters
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_4')(x)
    x = Dropout(0.2, name='dropout_4')(x)

    # Flatten & Dense
    x = Flatten(name='flatten')(x)
    x = Dense(128, activation='relu', name='dense_1')(x)
    x = Dropout(0.3, name='dropout_final')(x)

    # Output
    outputs = Dense(1, activation='linear', name='output_layer')(x)

    return Model(inputs=inputs, outputs=outputs, name='ShallowCNN')
