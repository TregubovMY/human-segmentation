import os

import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, ReLU
from keras.layers import BatchNormalization, Conv2DTranspose, Concatenate
from keras.models import Model
from keras.utils import CustomObjectScope
from ..metrics.metrics import *
from ..utils.utils import folder_path

# Сверточный слой
def convolution_operation(entered_input, filters=64):
    """Реализуем первый блок conv"""
    conv1 = Conv2D(filters, kernel_size=(3, 3), padding="same")(entered_input)
    batch_norm1 = BatchNormalization()(conv1)
    act1 = ReLU()(batch_norm1)

    """Реализуем второй блок conv"""
    conv2 = Conv2D(filters, kernel_size=(3, 3), padding="same")(act1)
    batch_norm2 = BatchNormalization()(conv2)
    act2 = ReLU()(batch_norm2)

    return act2


def encoder(entered_input, filters=64):
    """Вызываем двойную свертку, макспулинг и верном оба для последующего
    использования в decodere"""
    enc1 = convolution_operation(entered_input, filters)
    maxPool1 = MaxPooling2D(strides=(2, 2))(enc1)

    return enc1, maxPool1


def decoder(entered_input, skip, filters=64):
    """Повышаем дискретизацию и объединяем слои [[a11..a1n, b11..b1n],..]"""
    upsample = Conv2DTranspose(filters, kernel_size=(2, 2), strides=2, padding="same")(
        entered_input
    )
    connect_Skip = Concatenate()([upsample, skip])
    out = convolution_operation(connect_Skip, filters)

    return out


def U_Net(image_size = (512, 512, 3)):
    """Берем размеры и форму изображения"""
    input_1 = Input(image_size)

    """Создаем блоки энкодера"""
    skip_1, encoder_1 = encoder(input_1, 64)
    skip_2, encoder_2 = encoder(encoder_1, 64 * 2)
    skip_3, encoder_3 = encoder(encoder_2, 64 * 4)
    skip_4, encoder_4 = encoder(encoder_3, 64 * 8)

    """Создаем блок перед декодером"""
    conv_block = convolution_operation(encoder_4, 64 * 16)

    """Создаем блоки декодера"""
    decoder_1 = decoder(conv_block, skip_4, 64 * 8)
    decoder_2 = decoder(decoder_1, skip_3, 64 * 4)
    decoder_3 = decoder(decoder_2, skip_2, 64 * 2)
    decoder_4 = decoder(decoder_3, skip_1, 64)

    out = Conv2D(1, 1, padding="same", activation="sigmoid")(decoder_4)

    model = Model(input_1, out)

    return model


def model_U_Net():
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
