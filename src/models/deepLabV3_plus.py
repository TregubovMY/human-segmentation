import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from typing import Tuple

from keras.layers import (
    Conv2D,
    BatchNormalization,
    Activation,
    Concatenate,
    Input,
    AveragePooling2D,
    GlobalAveragePooling2D,
    UpSampling2D,
    Reshape,
    Dense,
)
from keras.applications import ResNet50
from keras.models import Model
import tensorflow as tf
from keras.utils import CustomObjectScope

from ..metrics.metrics import iou, dice_coef, dice_loss
from ..utils.utils import folder_path

def convolution_block(
    block_input: tf.Tensor,
    num_filters: int = 256,
    kernel_size: int = 3,
    dilation_rate: int = 1,
    padding: str = "same",
    use_bias: bool = False,
) -> tf.Tensor:
    """
    Создает сверточный блок.

    :param block_input: Входной тензор.
    :type block_input: tf.Tensor
    :param num_filters: Количество фильтров свертки. По умолчанию 256.
    :type num_filters: int, optional
    :param kernel_size: Размер ядра свертки. По умолчанию 3.
    :type kernel_size: int, optional
    :param dilation_rate: Шаг расширения свертки. По умолчанию 1.
    :type dilation_rate: int, optional
    :param padding: Тип заполнения. По умолчанию "same".
    :type padding: str, optional
    :param use_bias: Использовать ли смещение в свертке. По умолчанию False.
    :type use_bias: bool, optional
    :return: Выходной тензор.
    :rtype: tf.Tensor
    """
    x = Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding=padding,
        use_bias=use_bias,
    )(block_input)
    x = BatchNormalization()(x)
    return Activation("relu")(x)


def ASPP(dspp_input: tf.Tensor) -> tf.Tensor:
    """
    Реализует модуль Atrous Spatial Pyramid Pooling (ASPP).

    :param dspp_input: Входной тензор.
    :type dspp_input: tf.Tensor
    :return: Выходной тензор.
    :rtype: tf.Tensor
    """
    shape = dspp_input.shape
    x = AveragePooling2D(pool_size=(shape[1], shape[2]))(dspp_input)
    x = convolution_block(x, kernel_size=1)
    out_pool = UpSampling2D((shape[1], shape[2]), interpolation="bilinear")(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    out = convolution_block(x, kernel_size=1, dilation_rate=1)
    return out


def SqueezeAndExcite(inputs: tf.Tensor, ratio: int = 8) -> tf.Tensor:
    """
    Реализует блок Squeeze-and-Excitation (SE).

    :param inputs: Входной тензор.
    :type inputs: tf.Tensor
    :param ratio: Коэффициент сжатия. По умолчанию 8.
    :type ratio: int, optional
    :return: Выходной тензор.
    :rtype: tf.Tensor
    """
    init = inputs
    filters = init.shape[-1]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(
        filters // ratio,
        activation="relu",
        kernel_initializer="he_normal",
        use_bias=False,
    )(se)
    se = Dense(
        filters, activation="sigmoid", kernel_initializer="he_normal", use_bias=False
    )(se)
    x = init * se
    return x


def deepLabV3_plus(shape: Tuple[int, int, int]) -> Model:
    """
    Создает модель DeepLabV3+.

    :param shape: Форма входных данных (высота, ширина, каналы).
    :type shape: Tuple[int, int, int]
    :return: Модель DeepLabV3+.
    :rtype: Model
    """
    inputs = Input(shape)

    encoder = ResNet50(weights="imagenet", include_top=False, input_tensor=inputs)

    image_features = encoder.get_layer("conv4_block6_out").output
    input_a = ASPP(image_features)
    input_a = UpSampling2D((4, 4), interpolation="bilinear")(input_a)

    input_b = encoder.get_layer("conv2_block2_out").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = Concatenate()([input_a, input_b])
    x = SqueezeAndExcite(x)

    x = convolution_block(x)
    x = convolution_block(x)
    x = SqueezeAndExcite(x)

    x = UpSampling2D((4, 4), interpolation="bilinear")(x)
    x = Conv2D(1, 1)(x)  # filter 1

    x = Activation("sigmoid")(x)

    return Model(inputs, x)

def model_deepLabV3_plus() -> Model:
    """
    Загружает модель DeepLabV3+ из файла, если он существует, иначе создает новую модель.

    :return: Модель DeepLabV3+.
    :rtype: Model
    """
    model_name = "DeepLabV3_plus"
    model_path = os.path.join(folder_path(), "models", "deepLabV3_plus.h5")

    if os.path.exists(model_path):
        print(f"{model_name}: Файл с весами {model_name} найден.")
        with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
            model = tf.keras.models.load_model(model_path)
        print(f"{model_name}: Веса успешно загружены.")
        return model
    else:
        print(f"{model_name}: Файл с весами не найден: {model_path}")
        return deepLabV3_plus((512, 512, 3))