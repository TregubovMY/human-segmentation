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
    """Загружает пути к изображениям и маскам из указанной директории.

    Args:
        path (str): Путь к директории с данными, содержащей подпапки "images" и "masks".

    Returns:
        Tuple[List[str], List[str]]: Кортеж из двух списков: путей к изображениям и путей к маскам.
    """

    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    mask_extensions = ["*.png"]

    x = []
    for ext in image_extensions:
        x.extend(sorted(glob(os.path.join(path, "images", ext))))

    y = []
    for ext in mask_extensions:
        y.extend(sorted(glob(os.path.join(path, "masks", ext))))
    return x, y

def shuffling(x: List[str], y: List[str]) -> Tuple[List[str], List[str]]:
    """Перемешивает списки изображений и масок.

    Args:
        x (List[str]): Список путей к изображениям.
        y (List[str]): Список путей к маскам.

    Returns:
        Tuple[List[str], List[str]]: Кортеж из двух перемешанных списков: путей к изображениям и путей к маскам.
    """

    x, y = shuffle(x, y, random_state=42)
    return x, y

def split_dataset(path: str, split: float = 0.1) -> Tuple[Tuple[List[str], List[str]], Tuple[List[str], List[str]]]:
    """Разбивает набор данных на обучающую и тестовую выборки.

    Args:
        path (str): Путь к директории с данными.
        split (float, optional): Доля данных, выделяемая для тестовой выборки. По умолчанию 0.1.

    Returns:
        Tuple[Tuple[List[str], List[str]], Tuple[List[str], List[str]]]: Кортеж из двух кортежей:
            - Первый кортеж содержит списки путей к изображениям и маскам для обучения.
            - Второй кортеж содержит списки путей к изображениям и маскам для тестирования.
    """

    X, Y = load_data(path)

    split_size = int(len(X) * split)

    train_x, test_x = train_test_split(X, test_size=split_size, random_state=42)
    train_y, test_y = train_test_split(Y, test_size=split_size, random_state=42)

    return (train_x, train_y), (test_x, test_y)

def augment_data(images: List[str], masks: List[str], save_path: str, cfg: DictConfig, augment: bool = True) -> None:
    """
    Аугментация изображений и масок.

    Args:
        images (List[str]): Список путей к изображениям.
        masks (List[str]): Список путей к маскам.
        save_path (str): Путь к директории для сохранения аугментированных данных.
        cfg (DictConfig): Конфигурация аугментации.
        augment (bool, optional): Флаг, указывающий, нужно ли выполнять аугментацию. 
                                    По умолчанию True.
    """

    def apply_augmentation(image: np.ndarray, mask: np.ndarray, aug: Callable) -> Tuple[np.ndarray, np.ndarray]:
        """
        Применяет аугментацию к изображению и маске.

        Args:
            image (np.ndarray): Изображение.
            mask (np.ndarray): Маска.
            aug (Callable): Функция аугментации.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Аугментированные изображение и маска.
        """
        augmented = aug(image=image, mask=mask)
        return augmented["image"], augmented["mask"]

    def save_images(image: np.ndarray, mask: np.ndarray, name: str, index: int) -> None:
        """
        Сохраняет аугментированные изображения и маски.

        Args:
            image (np.ndarray): Изображение.
            mask (np.ndarray): Маска.
            name (str): Имя файла.
            index (int): Индекс аугментации.
        """
        temp_image_name = f"{name}_{index}.png"
        temp_mask_name = f"{name}_{index}.png"
        image_path = os.path.join(save_path, "images", temp_image_name)
        mask_path = os.path.join(save_path, "masks", temp_mask_name)
        cv2.imwrite(image_path, image)
        cv2.imwrite(mask_path, mask)

    def process_image(x: np.ndarray, y: np.ndarray, cfg: DictConfig) -> Tuple[np.ndarray, np.ndarray]:
        """
        Обрабатывает изображение и маску: изменение размера или кадрирование.

        Args:
            x (np.ndarray): Изображение.
            y (np.ndarray): Маска.
            cfg (DictConfig): Конфигурация.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Обработанные изображение и маска.
        """
        try:
            HEIGHT = cfg.resolution.HEIGHT
            WIDTH = cfg.resolution.WIDTH
            aug = CenterCrop(HEIGHT, WIDTH, p=1.0)
            augmentation = aug(image=x, mask=y)
            x, y = augmentation["image"], augmentation["mask"]
        except Exception as e:
            x = cv2.resize(x, (WIDTH, HEIGHT))
            y = cv2.resize(y, (WIDTH, HEIGHT))
        return x, y

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
