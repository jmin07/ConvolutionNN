import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense , Conv2D , Dropout , Flatten 
from tensorflow.keras.layers import Activation, MaxPooling2D
from tensorflow.keras.layers import ReLU
from tensorflow.nn import local_response_normalization

def AlexNet():
    inputs = Input(shape=(227, 227, 3))
        
    x = Conv2D(96, kernel_size=(11, 11), strides=4, padding='valid')(inputs)

    x = Activation('ReLU')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    x = local_response_normalization(x)

    x = Conv2D(256, kernel_size=(3, 3), padding='same')(x)

    x = Activation('ReLU')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    x = local_response_normalization(x)

    x = Conv2D(384, kernel_size=(3, 3), padding='same', strides=1)(x)
    x = Conv2D(384, kernel_size=(3, 3), padding='same', strides=1)(x)
    x = Conv2D(256, kernel_size=(1, 1), padding='same', strides=1)(x)

    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)

    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)

    outputs = Dense(1000, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model
