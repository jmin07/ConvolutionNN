def inception_module(x, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, fitlers_5x5, filters_pool_proj, name=None):
    """
      x: input_shape
      filters_3x3_reduce: filter_3x3 전 1x1 filter 의 output channel
      filters_5x5_reduce: fitler_5x5 전 1x1 filter 의 output channel
    """
    
    # 첫 번째 1x1 Conv
    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu')(x)

    # 3x3 적용 전 1x1 conv -> 3x3 conv
    conv_3x3 = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu')(x)
    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu')(conv_3x3)

    # 5x5 적용전 1x1 conv -> 5x5 conv
    conv_5x5 = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu')(x)
    conv_5x5 = Conv2D(fitlers_5x5, (5, 5), padding='same', activation='relu')(conv_5x5)

    # MaxPooluing2D
    pool_proj = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu')(pool_proj)

    # feature map을 channel 기준으로 concat 적용.
    ouput = Concatenate(axis=-1, name=name)([conv_1x1, conv_3x3, conv_5x5, pool_proj])

    return ouput

def GoogleNext(in_shape=(224, 224, 3), n_classes=10):
    input_tensor = Input(in_shape)

    x = Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', name='conv_7x7/2')(input_tensor)
    x = MaxPooling2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)
    x = Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu', name='conv_2a_3x3/1')(x)
    x = Conv2D(194, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv_2b_3x3/1')(x)
    x = MaxPooling2D((3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3/2')(x)

    # 첫 번재 inception 모듈
    x = inception_module(x, 
                filters_1x1=64,
                filters_3x3_reduce=96,
                filters_3x3=128,
                filters_5x5_reduce=16,
                fitlers_5x5=32,
                filters_pool_proj=32,
                name='inception_3a')

    # 두 번재 inception 모듈
    x = inception_module(x, 
                filters_1x1=128,
                filters_3x3_reduce=128,
                filters_3x3=192,
                filters_5x5_reduce=32,
                fitlers_5x5=96,
                filters_pool_proj=64,
                name='inception_3b')

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='max_pool_3_3x3/2')(x)

    # 세 번째 inception 모듈
    x = inception_module(x, 
                filters_1x1=192,
                filters_3x3_reduce=96,
                filters_3x3=208,
                filters_5x5_reduce=16,
                fitlers_5x5=48,
                filters_pool_proj=64,
                name='inception_4a')

    # 네 번째 inception 모듈
    x = inception_module(x, 
                filters_1x1=160,
                filters_3x3_reduce=112,
                filters_3x3=224,
                filters_5x5_reduce=24,
                fitlers_5x5=64,
                filters_pool_proj=64,
                name='inception_4b')

    # 다섯 번째 inception 모듈
    x = inception_module(x, 
                filters_1x1=128,
                filters_3x3_reduce=128,
                filters_3x3=256,
                filters_5x5_reduce=24,
                fitlers_5x5=64,
                filters_pool_proj=64,
                name='inception_4c')

    # 여섯 번째 inception 모듈
    x = inception_module(x, 
                filters_1x1=112,
                filters_3x3_reduce=144,
                filters_3x3=288,
                filters_5x5_reduce=32,
                fitlers_5x5=64,
                filters_pool_proj=64,
                name='inception_4d')
     # 일곱 번째 inception 모듈
    x = inception_module(x, 
                filters_1x1=256,
                filters_3x3_reduce=160,
                filters_3x3=320,
                filters_5x5_reduce=32,
                fitlers_5x5=128,
                filters_pool_proj=128,
                name='inception_4e')

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='max_pool_4_3x3/2')(x)

    # 여덟 번째 inception 모듈
    x = inception_module(x, 
                filters_1x1=256,
                filters_3x3_reduce=160,
                filters_3x3=320,
                filters_5x5_reduce=32,
                fitlers_5x5=128,
                filters_pool_proj=128,
                name='inception_5a')

    # 아홉 번째 inception 모듈
    x = inception_module(x, 
                filters_1x1=384,
                filters_3x3_reduce=192,
                filters_3x3=384,
                filters_5x5_reduce=48,
                fitlers_5x5=128,
                filters_pool_proj=128,
                name='inception_5b')

    x = GlobalAveragePooling2D(name='avg_pool_5_3x3/1')(x)
    x = Dropout(0.5)(x)

    output = Dense(n_classes, activation='softmax', name='output')(x)

    model = Model(inputs=input_tensor, outputs=output)

    return model
