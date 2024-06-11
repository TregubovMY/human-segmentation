import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from keras.layers import (
    Conv2D,
    BatchNormalization,
    Activation,
    Concatenate,
    Input,
)
from keras.layers import (
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
from ..metrics.metrics import *
from ..utils.utils import folder_path

def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding=padding,
        use_bias=use_bias,
    )(block_input)
    x = BatchNormalization()(x)
    return Activation("relu")(x)

def convolution_block(block_input, num_filters=256, kernel_size=3, dilation_rate=1, padding="same"):
    x = Conv2D(num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding)(block_input)
    x = BatchNormalization()(x)
    return Activation("relu")(x)


def ASPP(dspp_input):
    """Пулинг"""
    shape = dspp_input.shape
    x = AveragePooling2D(pool_size=(shape[1], shape[2]))(dspp_input)
    x = convolution_block(x, kernel_size=1)
    out_pool = UpSampling2D((shape[1], shape[2]), interpolation="bilinear")(x)

    """Свертки"""
    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    out = convolution_block(x, kernel_size=1, dilation_rate=1)
    return out


def SqueezeAndExcite(inputs, ratio=8):
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


def deepLabV3_plus(shape):
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

def model_deepLabV3_plus():
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
