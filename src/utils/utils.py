import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import tensorflow as tf
from omegaconf import DictConfig
from typing import Tuple

def create_dir(path: str) -> None:
    """
    Создает директорию, если она не существует.

    :param path: Путь к директории.
    :type path: str
    """
    if not os.path.exists(path):
        os.makedirs(path)

def read_image(path: bytes, cfg: DictConfig) -> np.ndarray:
    """
    Читает изображение, изменяет его размер (если необходимо) и нормализует.

    :param path: Путь к изображению в байтовом формате.
    :type path: bytes
    :param cfg: Конфигурация, содержащая информацию о модели и разрешении изображения.
    :type cfg: DictConfig
    :return: Нормализованное изображение в формате NumPy.
    :rtype: np.ndarray
    """
    path = path.decode()
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if cfg.model.name in ["u2_net", "u2_net_lite"]:
        image = cv2.resize(image, (cfg.resolution.WIDTH, cfg.resolution.HEIGHT))

    image = image / 255.0
    image = image.astype(np.float32)
    return image

def read_mask(path: bytes, cfg: DictConfig) -> np.ndarray:
    """
    Читает маску, изменяет ее размер (если необходимо) и нормализует.

    :param path: Путь к маске в байтовом формате.
    :type path: bytes
    :param cfg: Конфигурация, содержащая информацию о модели и разрешении изображения.
    :type cfg: DictConfig
    :return: Нормализованная маска в формате NumPy.
    :rtype: np.ndarray
    """
    path = path.decode()
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if cfg.model.name in ["u2_net", "u2_net_lite"]:
        mask = cv2.resize(mask, (cfg.resolution.WIDTH, cfg.resolution.HEIGHT))
        mask = mask / 255.0

    mask = mask.astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)
    return mask

def tf_parse(x: bytes, y: bytes, cfg: DictConfig) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Функция для парсинга изображения и маски с использованием TensorFlow.

    :param x: Путь к изображению в байтовом формате.
    :type x: bytes
    :param y: Путь к маске в байтовом формате.
    :type y: bytes
    :param cfg: Конфигурация, содержащая информацию о модели и разрешении изображения.
    :type cfg: DictConfig
    :return: Кортеж из двух тензоров: изображения и маски.
    :rtype: Tuple[tf.Tensor, tf.Tensor]
    """
    def _parse(x: bytes, y: bytes) -> Tuple[np.ndarray, np.ndarray]:
        """
        Вспомогательная функция для парсинга изображения и маски.

        :param x: Путь к изображению в байтовом формате.
        :type x: bytes
        :param y: Путь к маске в байтовом формате.
        :type y: bytes
        :return: Кортеж из двух NumPy массивов: изображения и маски.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        x = read_image(x, cfg)
        y = read_mask(y, cfg)
        return x, y

    HEIGHT = cfg.resolution.HEIGHT
    WIDTH = cfg.resolution.WIDTH
    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([HEIGHT, WIDTH, 3])
    y.set_shape([HEIGHT, WIDTH, 1])
    return x, y

def tf_dataset(X: np.ndarray, Y: np.ndarray, cfg: DictConfig, batch: int = 2) -> tf.data.Dataset:
    """
    Создает TensorFlow Dataset из путей к изображениям и маскам.

    :param X: Массив путей к изображениям.
    :type X: np.ndarray
    :param Y: Массив путей к маскам.
    :type Y: np.ndarray
    :param cfg: Конфигурация, содержащая информацию о модели и разрешении изображения.
    :type cfg: DictConfig
    :param batch: Размер батча. По умолчанию 2.
    :type batch: int, optional
    :return: TensorFlow Dataset.
    :rtype: tf.data.Dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(lambda x, y: tf_parse(x, y, cfg))
    dataset = dataset.batch(batch).prefetch(10)
    return dataset

def folder_path() -> str:
    """
    Возвращает путь к корневой директории проекта.

    :return: Путь к корневой директории проекта.
    :rtype: str
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    return base_dir

def save_results(image: np.ndarray, mask: np.ndarray, y_pred: np.ndarray, save_image_path: str) -> None:
    """
    Сохраняет результаты сегментации в виде изображения.

    :param image: Оригинальное изображение.
    :type image: np.ndarray
    :param mask: Истинная маска.
    :type mask: np.ndarray
    :param y_pred: Предсказанная маска.
    :type y_pred: np.ndarray
    :param save_image_path: Путь для сохранения изображения с результатами.
    :type save_image_path: str
    """
    height = image.shape[0]
    line = np.ones((height, 10, 3)) * 128

    mask = np.expand_dims(mask, axis=-1)    
    mask = np.concatenate([mask, mask, mask], axis=-1)  

    y_pred = np.expand_dims(y_pred, axis=-1)    
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)

    masked_image = image * y_pred
    y_pred = y_pred * 255

    concatenated_images = np.concatenate([image, line, mask, line, y_pred, line, masked_image], axis=1)
    cv2.imwrite(save_image_path, concatenated_images)
