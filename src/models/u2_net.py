import os
from typing import Tuple, List

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
from keras.utils import CustomObjectScope

from src.utils.utils import folder_path
from ..metrics.metrics import iou, dice_coef, dice_loss

def conv_block(inputs: tf.Tensor, out_ch: int, rate: int = 1) -> tf.Tensor:
    """
    Создает сверточный блок с BatchNormalization и ReLU активацией.

    :param inputs: Входной тензор.
    :type inputs: tf.Tensor
    :param out_ch: Количество выходных каналов.
    :type out_ch: int
    :param rate: Коэффициент расширения (dilation rate) для свертки. По умолчанию 1.
    :type rate: int, optional
    :return: Выходной тензор.
    :rtype: tf.Tensor
    """
    x = Conv2D(out_ch, 3, padding="same", dilation_rate=rate)(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def RSU_L(inputs: tf.Tensor, out_ch: int, int_ch: int, num_layers: int, rate: int = 2) -> tf.Tensor:
    """
    Реализует блок Residual U-shaped network (RSU) типа "L".

    :param inputs: Входной тензор.
    :type inputs: tf.Tensor
    :param out_ch: Количество выходных каналов.
    :type out_ch: int
    :param int_ch: Количество промежуточных каналов.
    :type int_ch: int
    :param num_layers: Количество слоев в энкодере и декодере.
    :type num_layers: int
    :param rate: Коэффициент расширения (dilation rate) для свертки в блоке Bridge. 
                 По умолчанию 2.
    :type rate: int, optional
    :return: Выходной тензор.
    :rtype: tf.Tensor
    """
    x = conv_block(inputs, out_ch)
    init_feats = x

    skip = []
    x = conv_block(x, int_ch)
    skip.append(x)

    for _ in range(num_layers - 2):
        x = MaxPool2D((2, 2))(x)
        x = conv_block(x, int_ch)
        skip.append(x)

    x = conv_block(x, int_ch, rate=rate)

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

    x = Add()([x, init_feats])
    return x

def RSU_4F(inputs: tf.Tensor, out_ch: int, int_ch: int) -> tf.Tensor:
    """
    Реализует блок Residual U-shaped network (RSU) типа "4F".

    :param inputs: Входной тензор.
    :type inputs: tf.Tensor
    :param out_ch: Количество выходных каналов.
    :type out_ch: int
    :param int_ch: Количество промежуточных каналов.
    :type int_ch: int
    :return: Выходной тензор.
    :rtype: tf.Tensor
    """
    x0 = conv_block(inputs, out_ch, rate=1)

    x1 = conv_block(x0, int_ch, rate=1)
    x2 = conv_block(x1, int_ch, rate=2)
    x3 = conv_block(x2, int_ch, rate=4)

    x4 = conv_block(x3, int_ch, rate=8)

    x = Concatenate()([x4, x3])
    x = conv_block(x, int_ch, rate=4)

    x = Concatenate()([x, x2])
    x = conv_block(x, int_ch, rate=2)

    x = Concatenate()([x, x1])
    x = conv_block(x, out_ch, rate=1)

    x = Add()([x, x0])
    return x

def u2net(
    input_shape: Tuple[int, int, int],
    out_ch: List[int],
    int_ch: List[int],
    num_classes: int = 1,
) -> Model:
    """
    Создает модель U2-Net для сегментации изображений.

    :param input_shape: Размер входного изображения (высота, ширина, каналы).
    :type input_shape: Tuple[int, int, int]
    :param out_ch: Список количеств выходных каналов для каждого блока RSU.
    :type out_ch: List[int]
    :param int_ch: Список количеств промежуточных каналов для каждого блока RSU.
    :type int_ch: List[int]
    :param num_classes: Количество классов для сегментации. По умолчанию 1.
    :type num_classes: int, optional
    :return: Модель U2-Net.
    :rtype: Model
    """
    inputs = Input(input_shape)
    s0 = inputs

    s1 = RSU_L(s0, out_ch[0], int_ch[0], 7)
    p1 = MaxPool2D((2, 2))(s1)

    s2 = RSU_L(p1, out_ch[1], int_ch[1], 6)
    p2 = MaxPool2D((2, 2))(s2)

    s3 = RSU_L(p2, out_ch[2], int_ch[2], 5)
    p3 = MaxPool2D((2, 2))(s3)

    s4 = RSU_L(p3, out_ch[3], int_ch[3], 4)
    p4 = MaxPool2D((2, 2))(s4)

    s5 = RSU_4F(p4, out_ch[4], int_ch[4])
    p5 = MaxPool2D((2, 2))(s5)

    b1 = RSU_4F(p5, out_ch[5], int_ch[5])
    b2 = UpSampling2D(size=(2, 2), interpolation="bilinear")(b1)

    d1 = Concatenate()([b2, s5])
    d1 = RSU_4F(d1, out_ch[6], int_ch[6])
    u1 = UpSampling2D(size=(2, 2), interpolation="bilinear")(d1)

    d2 = Concatenate()([u1, s4])
    d2 = RSU_L(d2, out_ch[7], int_ch[7], 4)
    u2 = UpSampling2D(size=(2, 2), interpolation="bilinear")(d2)

    d3 = Concatenate()([u2, s3])
    d3 = RSU_L(d3, out_ch[8], int_ch[8], 5)
    u3 = UpSampling2D(size=(2, 2), interpolation="bilinear")(d3)

    d4 = Concatenate()([u3, s2])
    d4 = RSU_L(d4, out_ch[9], int_ch[9], 6)
    u4 = UpSampling2D(size=(2, 2), interpolation="bilinear")(d4)

    d5 = Concatenate()([u4, s1])
    d5 = RSU_L(d5, out_ch[10], int_ch[10], 7)

    y1 = Conv2D(num_classes, 3, padding="same")(d5)

    y2 = Conv2D(num_classes, 3, padding="same")(d4)
    y2 = UpSampling2D(size=(2, 2), interpolation="bilinear")(y2)

    y3 = Conv2D(num_classes, 3, padding="same")(d3)
    y3 = UpSampling2D(size=(4, 4), interpolation="bilinear")(y3)

    y4 = Conv2D(num_classes, 3, padding="same")(d2)
    y4 = UpSampling2D(size=(8, 8), interpolation="bilinear")(y4)

    y5 = Conv2D(num_classes, 3, padding="same")(d1)
    y5 = UpSampling2D(size=(16, 16), interpolation="bilinear")(y5)

    y6 = Conv2D(num_classes, 3, padding="same")(b1)
    y6 = UpSampling2D(size=(32, 32), interpolation="bilinear")(y6)

    y0 = Concatenate()([y1, y2, y3, y4, y5, y6])
    y0 = Conv2D(num_classes, 3, padding="same")(y0)

    y0 = Activation("sigmoid", name="y0")(y0)
    y1 = Activation("sigmoid", name="y1")(y1)
    y2 = Activation("sigmoid", name="y2")(y2)
    y3 = Activation("sigmoid", name="y3")(y3)
    y4 = Activation("sigmoid", name="y4")(y4)
    y5 = Activation("sigmoid", name="y5")(y5)
    y6 = Activation("sigmoid", name="y6")(y6)

    model = tf.keras.models.Model(inputs, outputs=[y0, y1, y2, y3, y4, y5, y6])
    return model

def build_u2net(input_shape: Tuple[int, int, int] = (512, 512, 3), num_classes: int = 1) -> Model:
    """
    Создает модель U2-Net с предопределенными параметрами.

    :param input_shape: Размер входного изображения (высота, ширина, каналы). 
                        По умолчанию (512, 512, 3).
    :type input_shape: Tuple[int, int, int], optional
    :param num_classes: Количество классов для сегментации. По умолчанию 1.
    :type num_classes: int, optional
    :return: Модель U2-Net.
    :rtype: Model
    """
    out_ch = [64, 128, 256, 512, 512, 512, 512, 256, 128, 64, 64]
    int_ch = [32, 32, 64, 128, 256, 256, 256, 128, 64, 32, 16]
    model = u2net(input_shape, out_ch, int_ch, num_classes=num_classes)
    return model

def build_u2net_lite(input_shape: Tuple[int, int, int] = (256, 256, 3), num_classes: int = 1) -> Model:
    """
    Создает облегченную модель U2-Net Lite с предопределенными параметрами.

    :param input_shape: Размер входного изображения (высота, ширина, каналы). 
                        По умолчанию (256, 256, 3).
    :type input_shape: Tuple[int, int, int], optional
    :param num_classes: Количество классов для сегментации. По умолчанию 1.
    :type num_classes: int, optional
    :return: Модель U2-Net Lite.
    :rtype: Model
    """
    out_ch = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
    int_ch = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]
    model = u2net(input_shape, out_ch, int_ch, num_classes=num_classes)
    return model


def model_U2_Net() -> Model:
    """
    Загружает модель U2-Net из файла, если он существует, 
    иначе создает новую модель.

    :return: Модель U2-Net.
    :rtype: Model
    """
    model_name = "U2-Net"
    model_path = os.path.join(folder_path(), "models", "u2_net.h5")

    if os.path.exists(model_path):
        print(f"{model_name}: Файл с весами {model_name} найден.")
        with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
            model = tf.keras.models.load_model(model_path)
        print(f"{model_name}: Веса успешно загружены.")
        return model
    else:
        print(f"{model_name}: Файл с весами не найден: {model_path}")
        return build_u2net()

def model_U2_Net_lite() -> Model:
    """
    Загружает модель U2-Net Lite из файла, если он существует, 
    иначе создает новую модель.

    :return: Модель U2-Net Lite.
    :rtype: Model
    """
    model_name = "U2-Net-Lite"
    model_path = os.path.join(folder_path(), "models", "u2_net_lite.h5")

    if os.path.exists(model_path):
        print(f"{model_name}: Файл с весами {model_name} найден.")
        with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
            model = tf.keras.models.load_model(model_path)
        print(f"{model_name}: Веса успешно загружены.")
        return model
    else:
        print(f"{model_name}: Файл с весами не найден: {model_path}")
        return build_u2net_lite()
