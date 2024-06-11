import os
from typing import Tuple

import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, ReLU
from keras.layers import BatchNormalization, Conv2DTranspose, Concatenate
from keras.models import Model
from keras.utils import CustomObjectScope

from ..metrics.metrics import iou, dice_coef, dice_loss
from ..utils.utils import folder_path

def convolution_operation(entered_input: tf.Tensor, filters: int = 64) -> tf.Tensor:
    """
    Реализует двойной сверточный блок с BatchNormalization и ReLU активацией.

    :param entered_input: Входной тензор.
    :type entered_input: tf.Tensor
    :param filters: Количество фильтров в сверточных слоях. По умолчанию 64.
    :type filters: int, optional
    :return: Выходной тензор после двойной свертки.
    :rtype: tf.Tensor
    """
    conv1 = Conv2D(filters, kernel_size=(3, 3), padding="same")(entered_input)
    batch_norm1 = BatchNormalization()(conv1)
    act1 = ReLU()(batch_norm1)

    conv2 = Conv2D(filters, kernel_size=(3, 3), padding="same")(act1)
    batch_norm2 = BatchNormalization()(conv2)
    act2 = ReLU()(batch_norm2)

    return act2


def encoder(entered_input: tf.Tensor, filters: int = 64) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Реализует блок энкодера U-Net.

    :param entered_input: Входной тензор.
    :type entered_input: tf.Tensor
    :param filters: Количество фильтров в сверточном блоке. По умолчанию 64.
    :type filters: int, optional
    :return: Кортеж из двух тензоров: выход сверточного блока и выход MaxPooling.
    :rtype: Tuple[tf.Tensor, tf.Tensor]
    """
    enc = convolution_operation(entered_input, filters)
    max_pool = MaxPooling2D(strides=(2, 2))(enc)

    return enc, max_pool


def decoder(entered_input: tf.Tensor, skip: tf.Tensor, filters: int = 64) -> tf.Tensor:
    """
    Реализует блок декодера U-Net.

    :param entered_input: Входной тензор.
    :type entered_input: tf.Tensor
    :param skip: Тензор skip-соединения из энкодера.
    :type skip: tf.Tensor
    :param filters: Количество фильтров в сверточном блоке. По умолчанию 64.
    :type filters: int, optional
    :return: Выходной тензор после декодирования.
    :rtype: tf.Tensor
    """
    upsample = Conv2DTranspose(filters, kernel_size=(2, 2), strides=2, padding="same")(
        entered_input
    )
    connect_skip = Concatenate()([upsample, skip])
    out = convolution_operation(connect_skip, filters)

    return out


def U_Net(image_size: Tuple[int, int, int] = (512, 512, 3)) -> Model:
    """
    Создает модель U-Net для сегментации изображений.

    :param image_size: Размер входного изображения (высота, ширина, каналы). 
                       По умолчанию (512, 512, 3).
    :type image_size: Tuple[int, int, int], optional
    :return: Модель U-Net.
    :rtype: Model
    """
    input_layer = Input(image_size)

    skip_1, encoder_1 = encoder(input_layer, 64)
    skip_2, encoder_2 = encoder(encoder_1, 64 * 2)
    skip_3, encoder_3 = encoder(encoder_2, 64 * 4)
    skip_4, encoder_4 = encoder(encoder_3, 64 * 8)

    conv_block = convolution_operation(encoder_4, 64 * 16)

    decoder_1 = decoder(conv_block, skip_4, 64 * 8)
    decoder_2 = decoder(decoder_1, skip_3, 64 * 4)
    decoder_3 = decoder(decoder_2, skip_2, 64 * 2)
    decoder_4 = decoder(decoder_3, skip_1, 64)

    out = Conv2D(1, 1, padding="same", activation="sigmoid")(decoder_4)

    model = Model(input_layer, out)

    return model


def model_U_Net() -> Model:
    """
    Загружает модель U-Net из файла, если он существует, иначе создает новую модель.

    :return: Модель U-Net.
    :rtype: Model
    """
    model_name = "U-Net"
    model_path = os.path.join(folder_path(), "models", "u_net.h5")

    if os.path.exists(model_path):
        print(f"{model_name}: Файл с весами {model_name} найден.")
        with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
             model = tf.keras.models.load_model(model_path)
        print(f"{model_name}: Веса успешно загружены.")
        return model

    else:
        print(f"{model_name}: Файл с весами не найден: {model_path}")
        return U_Net()
