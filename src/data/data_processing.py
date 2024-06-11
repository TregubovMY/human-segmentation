import os
from glob import glob
from typing import Tuple, List, Callable

import cv2
import numpy as np
from albumentations import (
    HorizontalFlip,
    ChannelShuffle,
    CoarseDropout,
    CenterCrop,
    Rotate,
)
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm

def load_data(path: str) -> Tuple[List[str], List[str]]:
    """
    Загружает пути к изображениям и маскам из указанной директории.

    :param path: Путь к директории с данными, содержащей подпапки "images" и "masks".
    :type path: str
    :return: Кортеж из двух списков: путей к изображениям и путей к маскам.
    :rtype: Tuple[List[str], List[str]]
    """

    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    mask_extensions = ["*.png"]

    images = []
    for ext in image_extensions:
        images.extend(sorted(glob(os.path.join(path, "images", ext))))

    masks = []
    for ext in mask_extensions:
        masks.extend(sorted(glob(os.path.join(path, "masks", ext))))

    return images, masks

def shuffling(images: List[str], masks: List[str]) -> Tuple[List[str], List[str]]:
    """
    Перемешивает списки изображений и масок.

    :param images: Список путей к изображениям.
    :type images: List[str]
    :param masks: Список путей к маскам.
    :type masks: List[str]
    :return: Кортеж из двух перемешанных списков: путей к изображениям и путей к маскам.
    :rtype: Tuple[List[str], List[str]]
    """

    images, masks = shuffle(images, masks, random_state=42)
    return images, masks

def split_dataset(path: str, split: float = 0.1) -> Tuple[Tuple[List[str], List[str]], Tuple[List[str], List[str]]]:
    """
    Разбивает набор данных на обучающую и тестовую выборки.

    :param path: Путь к директории с данными.
    :type path: str
    :param split: Доля данных, выделяемая для тестовой выборки. По умолчанию 0.1.
    :type split: float, optional
    :return: Кортеж из двух кортежей:
        - Первый кортеж содержит списки путей к изображениям и маскам для обучения.
        - Второй кортеж содержит списки путей к изображениям и маскам для тестирования.
    :rtype: Tuple[Tuple[List[str], List[str]], Tuple[List[str], List[str]]]
    """

    images, masks = load_data(path)

    split_size = int(len(images) * split)

    train_images, test_images = train_test_split(images, test_size=split_size, random_state=42)
    train_masks, test_masks = train_test_split(masks, test_size=split_size, random_state=42)

    return (train_images, train_masks), (test_images, test_masks)

def augment_data(images: List[str], masks: List[str], save_path: str, cfg: DictConfig, augment: bool = True) -> None:
    """
    Аугментация изображений и масок.

    :param images: Список путей к изображениям.
    :type images: List[str]
    :param masks: Список путей к маскам.
    :type masks: List[str]
    :param save_path: Путь к директории для сохранения аугментированных данных.
    :type save_path: str
    :param cfg: Конфигурация аугментации.
    :type cfg: DictConfig
    :param augment: Флаг, указывающий, нужно ли выполнять аугментацию. 
                    По умолчанию True.
    :type augment: bool, optional
    """

    def apply_augmentation(image: np.ndarray, mask: np.ndarray, augmentation: Callable) -> Tuple[np.ndarray, np.ndarray]:
        """
        Применяет аугментацию к изображению и маске.

        :param image: Изображение.
        :type image: np.ndarray
        :param mask: Маска.
        :type mask: np.ndarray
        :param augmentation: Функция аугментации.
        :type augmentation: Callable
        :return: Аугментированные изображение и маска.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        augmented = augmentation(image=image, mask=mask)
        return augmented["image"], augmented["mask"]

    def save_images(image: np.ndarray, mask: np.ndarray, name: str, index: int) -> None:
        """
        Сохраняет аугментированные изображения и маски.

        :param image: Изображение.
        :type image: np.ndarray
        :param mask: Маска.
        :type mask: np.ndarray
        :param name: Имя файла.
        :type name: str
        :param index: Индекс аугментации.
        :type index: int
        """
        temp_image_name = f"{name}_{index}.png"
        temp_mask_name = f"{name}_{index}.png"
        image_path = os.path.join(save_path, "images", temp_image_name)
        mask_path = os.path.join(save_path, "masks", temp_mask_name)
        cv2.imwrite(image_path, image)
        cv2.imwrite(mask_path, mask)

    def process_image(image: np.ndarray, mask: np.ndarray, cfg: DictConfig) -> Tuple[np.ndarray, np.ndarray]:
        """
        Обрабатывает изображение и маску: изменение размера или кадрирование.

        :param image: Изображение.
        :type image: np.ndarray
        :param mask: Маска.
        :type mask: np.ndarray
        :param cfg: Конфигурация.
        :type cfg: DictConfig
        :return: Обработанные изображение и маска.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        try:
            height = cfg.resolution.HEIGHT
            width = cfg.resolution.WIDTH
            crop_augmentation = CenterCrop(height, width, p=1.0)
            augmentation = crop_augmentation(image=image, mask=mask)
            image, mask = augmentation["image"], augmentation["mask"]
        except Exception as e:
            image = cv2.resize(image, (width, height))
            mask = cv2.resize(mask, (width, height))
        return image, mask

    for x, y in tqdm(zip(images, masks), total=len(images)):
        nameFile = os.path.basename(x).split(".")[0]
        x, y = cv2.imread(x, cv2.IMREAD_COLOR), cv2.imread(y, cv2.IMREAD_COLOR)

        if augment:
            aug_1 = HorizontalFlip(p=1)
            x1, y1 = apply_augmentation(x, y, aug_1)

            x2, y2 = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), y

            aug_3 = ChannelShuffle(p=1)
            x3, y3 = apply_augmentation(x, y, aug_3)

            aug_4 = CoarseDropout(
                p=1, min_holes=3, max_holes=10, max_height=32, max_width=32
            )
            x4, y4 = apply_augmentation(x, y, aug_4)

            aug_5 = Rotate(limit=45, p=1.0)
            x5, y5 = apply_augmentation(x, y, aug_5)

            X = [x, x1, x2, x3, x4, x5]
            Y = [y, y1, y2, y3, y4, y5]

        else:
            X = [x]
            Y = [y]

        index = 0
        for image, mask in zip(X, Y):
            image, mask = process_image(image, mask, cfg)
            save_images(image, mask, nameFile, index)
            index += 1
