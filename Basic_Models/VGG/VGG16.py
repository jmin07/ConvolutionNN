from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D

def VGG(in_shape=(227,227,3), n_classes=10):
  input_tensor = Input(shape=in_shape)

  # Block 1
  x = Conv2D(64, (3, 3), activation='relu', padding='same', nmae='block1_conv1')(input_tensor)
  x = Conv2D(64, (3, 3), activation='relu', padding='same', nmae='block1_conv1')(x)
  x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block1_pool')(x)

  # Block 2
  x = Conv2D(128, (3, 3), activation='relu', padding='same', nmae='block2_conv1')(x)
  x = Conv2D(128, (3, 3), activation='relu', padding='same', nmae='block2_conv2')(x)
  x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block2_pool')(x)

  # Block 3
  x = Conv2D(256, (3, 3), activation='relu', padding='same', nmae='block3_conv1')(x)
  x = Conv2D(256, (3, 3), activation='relu', padding='same', nmae='block3_conv2')(x)
  x = Conv2D(256, (3, 3), activation='relu', padding='same', nmae='block3_conv3')(x)
  x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block3_pool')(x)

  # Block 4
  x = Conv2D(512, (3, 3), activation='relu', padding='same', nmae='block4_conv1')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', nmae='block4_conv2')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', nmae='block4_conv3')(x)
  x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block4_pool')(x)

  # Block 5
  x = Conv2D(512, (3, 3), activation='relu', padding='same', nmae='block5_conv1')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', nmae='block5_conv2')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', nmae='block5_conv3')(x)
  x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block5_pool')(x)

  x = GlobalAveragePooling2D()(x)
  x = Dropout(0.5)(x)
  x = Dense(120, activation='relu')(x)

  output = Dense(n_classes, activation='softmax')(x)

  model = Model(inputs=inputs, ouputs=outputs)

  return model


# block version 
def conv_block(tensor_in, filters, kernel_size, repeats=2, pool_strides=(2, 2), block_id):
    """Argument
        tensor_in: 입력 이미지 tensor
        filters: filter 개수
        kernel_size: kernel 크기
        repeats: conv 연산 회수(Layer 개수)
    """

    x = tensor_in

    for i in range(repeats):
        conv_name = 'block' + str(block_id) + '_conv' +str(i+1)
        x = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same', name=conv_name)(x)

    # max pooling 적용하여 출력 feature map의 크기를 절반으로 줄임
    x = MaxPooling2D(pool_size=(2, 2), strides=pool_strides, name='block' + str(block_id) + '_pool')(x)

    return x


def VGG_block(input_shape=(224,224,3), n_classes=10):

    inputs = Input(shape=input_shape, name='Input Tensor')

    # Block 1
    x = conv_block(inputs, filters=64, kernel_size=(3, 3), repeats=2, pool_strides=(2,2), block_id=1)

    # Block 2
    x = conv_block(x, filters=128, kernel_size=(3, 3), repeats=2, pool_strides=(2,2), block_id=2)
    
    # Block 3
    x = conv_block(x, filters=256, kernel_size=(3, 3), repeats=2, pool_strides=(2,2), block_id=3)

    # Block 4
    x = conv_block(x, filters=512, kernel_size=(3, 3), repeats=2, pool_strides=(2,2), block_id=4)

    # Block 5
    x = conv_block(x, filters=512, kernel_size=(3, 3), repeats=2, pool_strides=(2,2), block_id=4)

    # GlobalAveragePooling 으로 Flatten 적용.
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(120, activation='relu')(x)
    x = Dropout(0.5)(x)

    outputs = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs, name='vgg_by_blockl')

    return  model
