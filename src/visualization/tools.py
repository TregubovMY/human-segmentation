import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from omegaconf import DictConfig
import requests


def download_and_read_image(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            # Чтение изображения в формате numpy array
            image = np.asarray(bytearray(response.content), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            return image
        else:
            print(
                "Не удалось загрузить изображение. Код состояния:", response.status_code
            )
            return None
    except Exception as e:
        print("Произошла ошибка:", e)
        return None
    

def save(image, save_dir_name):
    cv2.imwrite(save_dir_name, image)


def predict(image, model, cfg: DictConfig):
    HEIGHT = cfg.resolution.HEIGHT
    WIDTH = cfg.resolution.WIDTH
    h, w, _ = image.shape
    x = cv2.resize(image, (WIDTH, HEIGHT))
    x = x / 255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)

    """ Предсказание """
    y = model.predict(x, verbose=0)

    # Проверка, является ли модель U2-Net
    if isinstance(y, list):
        y = y[0][0]
    else:
        y = y[0]

    y = cv2.resize(y, (w, h))
    y = np.expand_dims(y, axis=-1)

    return y


def masked_image(image, y):
    h, _, _ = image.shape
    masked_image = image * y

    return masked_image


def result_image_with_mask(image, masked_image):
    h, _, _ = image.shape
    line = np.ones((h, 10, 3)) * 255
    cat_images = np.concatenate([image, line, masked_image], axis=1)

    return cat_images

