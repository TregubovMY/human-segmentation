import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPool2D,
    UpSampling2D,
    Concatenate,
    Add,
)
from keras.models import Model
from src.utils.utils import folder_path
import os


def conv_block(inputs, out_ch, rate=1):
    x = Conv2D(out_ch, 3, padding="same", dilation_rate=rate)(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def RSU_L(inputs, out_ch, int_ch, num_layers, rate=2):
    """Начальная свертка Conv"""
    x = conv_block(inputs, out_ch)
    init_feats = x

    """ Энкодер """
    skip = []
    x = conv_block(x, int_ch)
    skip.append(x)

    for i in range(num_layers - 2):
        x = MaxPool2D((2, 2))(x)
        x = conv_block(x, int_ch)
        skip.append(x)

    """ Мост """
    x = conv_block(x, int_ch, rate=rate)

    """ Декодер """
    skip.reverse()

    x = Concatenate()([x, skip[0]])
    x = conv_block(x, int_ch)

    for i in range(num_layers - 3):
        x = UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
        x = Concatenate()([x, skip[i + 1]])
        x = conv_block(x, int_ch)

    x = UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
    x = Concatenate()([x, skip[-1]])
    x = conv_block(x, out_ch)

    """ Add """
    x = Add()([x, init_feats])
    return x


def RSU_4F(inputs, out_ch, int_ch):
    """Начальная свертка Conv"""
    x0 = conv_block(inputs, out_ch, rate=1)

    """ Энкодер """
    x1 = conv_block(x0, int_ch, rate=1)
    x2 = conv_block(x1, int_ch, rate=2)
    x3 = conv_block(x2, int_ch, rate=4)

    """ Мост """
    x4 = conv_block(x3, int_ch, rate=8)

    """ Декодер """
    x = Concatenate()([x4, x3])
    x = conv_block(x, int_ch, rate=4)

    x = Concatenate()([x, x2])
    x = conv_block(x, int_ch, rate=2)

    x = Concatenate()([x, x1])
    x = conv_block(x, out_ch, rate=1)

    """ Addition """
    x = Add()([x, x0])
    return x


def RSU_block(x, out_ch, int_ch, num_blocks):
    if num_blocks == 3:
        return RSU_4F(x, out_ch, int_ch)
    else:
        return RSU_L(x, out_ch, int_ch, num_blocks)


def concatenate_and_rsu(u, s, out_ch, int_ch, num_blocks):
    d = Concatenate()([u, s])
    d = RSU_block(d, out_ch, int_ch, num_blocks)
    return d


def u2net(input_shape, out_ch, int_ch, num_classes=1):
    """Входной слой"""
    inputs = Input(input_shape)

    """ Энкодер """
    s = []
    pool_layer = inputs

    for i in range(5):
        s.append(RSU_block(pool_layer, out_ch[i], int_ch[i], 7 - i))
        pool_layer = MaxPool2D((2, 2))(s[i])

    """ Мост """
    b1 = RSU_4F(pool_layer, out_ch[5], int_ch[5])
    b2 = UpSampling2D(size=(2, 2), interpolation="bilinear")(b1)

    """ Декодер """
    upsample_layers = []
    u = b2
    for i in range(5):
        upsample_layers.append(
            concatenate_and_rsu(u, s[-1 - i], out_ch[6 + i], int_ch[6 + i], 3 + i)
        )
        u = UpSampling2D(size=(2, 2), interpolation="bilinear")(upsample_layers[i])

    upsample_layers = [b1] + upsample_layers

    """ Выходной слой """
    y = Conv2D(num_classes, 3, padding="same")(upsample_layers[-1])
    side_outputs = [y]

    for i in range(1, 6):
        y = Conv2D(num_classes, 3, padding="same")(upsample_layers[-1 - i])
        y = UpSampling2D(size=(2**i, 2**i), interpolation="bilinear")(y)
        side_outputs.append(y)

    """ Финальный слой """
    final_output = Concatenate()(side_outputs)
    final_output = Conv2D(num_classes, 3, padding="same")(final_output)
    final_output = Activation("sigmoid")(final_output)

    side_outputs = [Activation("sigmoid")(layer) for layer in side_outputs]

    return Model(inputs, outputs=final_output)

def build_u2net(input_shape = (512, 512, 3), num_classes=1):
    out_ch = [64, 128, 256, 512, 512, 512, 512, 256, 128, 64, 64]
    int_ch = [32, 32, 64, 128, 256, 256, 256, 128, 64, 32, 16]
    model = u2net(input_shape, out_ch, int_ch, num_classes=num_classes)
    return model

def build_u2net_lite(input_shape = (512, 512, 3), num_classes=1):
    out_ch = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
    int_ch = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]
    model = u2net(input_shape, out_ch, int_ch, num_classes=num_classes)
    return model


def model_U2_Net():
    model_path = os.path.join(folder_path(), "models", "model_u2_net.h5")
    model = build_u2net()
    model.load_weights(model_path)

    return model