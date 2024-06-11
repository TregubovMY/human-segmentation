import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from omegaconf import DictConfig
import requests
from typing import Optional
import tensorflow as tf

def download_and_read_image(url: str) -> Optional[np.ndarray]:
    """
    Загружает изображение по URL и преобразует его в массив NumPy.

    :param url: URL изображения.
    :type url: str
    :return: Изображение в виде массива NumPy или None, если загрузка не удалась.
    :rtype: Optional[np.ndarray]
    """
    try:
        response = requests.get(url)
        if response.status_code == 200:
            image = np.asarray(bytearray(response.content), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            return image
        else:
            print(f"Не удалось загрузить изображение. Код состояния: {response.status_code}")
            return None
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        return None
    

def save(image: np.ndarray, save_path: str) -> None:
    """
    Сохраняет изображение в файл.

    :param image: Изображение для сохранения.
    :type image: np.ndarray
    :param save_path: Путь к файлу для сохранения.
    :type save_path: str
    """
    cv2.imwrite(save_path, image)


def predict(image: np.ndarray, model: tf.keras.Model, cfg: DictConfig) -> np.ndarray:
    """
    Выполняет предсказание маски для изображения с использованием заданной модели.

    :param image: Изображение для сегментации.
    :type image: np.ndarray
    :param model: Модель для сегментации.
    :type model: tf.keras.Model
    :param cfg: Конфигурация, содержащая информацию о моделях и разрешении изображения.
    :type cfg: DictConfig
    :return: Предсказанная маска.
    :rtype: np.ndarray
    """
    height = cfg.resolution.HEIGHT
    width = cfg.resolution.WIDTH
    original_height, original_width, _ = image.shape
    image = cv2.resize(image, (width, height))
    image = image / 255.0
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)

    """ Предсказание """
    prediction = model.predict(image, verbose=0)

    # Проверка, является ли модель U2-Net
    if isinstance(prediction, list):
        prediction = prediction[0][0]
    else:
        prediction = prediction[0]

    prediction = cv2.resize(prediction, (original_width, original_height))
    prediction = np.expand_dims(prediction, axis=-1)

    return prediction


def masked_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Создает изображение с наложенной маской.

    :param image: Оригинальное изображение.
    :type image: np.ndarray
    :param mask: Маска для наложения.
    :type mask: np.ndarray
    :return: Изображение с наложенной маской.
    :rtype: np.ndarray
    """
    masked_image = image * mask
    return masked_image


def result_image_with_mask(image: np.ndarray, masked_image: np.ndarray) -> np.ndarray:
    """
    Создает изображение, содержащее исходное изображение, разделительную линию и изображение с наложенной маской.

    :param image: Оригинальное изображение.
    :type image: np.ndarray
    :param masked_image: Изображение с наложенной маской.
    :type masked_image: np.ndarray
    :return: Изображение, содержащее исходное изображение, разделительную линию и изображение с наложенной маской.
    :rtype: np.ndarray
    """
    height, _, _ = image.shape
    line = np.ones((height, 10, 3)) * 255
    concatenated_images = np.concatenate([image, line, masked_image], axis=1)
    return concatenated_images
