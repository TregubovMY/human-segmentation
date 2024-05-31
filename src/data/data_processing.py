import os
from glob import glob
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split
from albumentations import (
    HorizontalFlip,
    ChannelShuffle,
    CoarseDropout,
    CenterCrop,
    Rotate,
)
from omegaconf import DictConfig
from sklearn.utils import shuffle

def load_data(path):
    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    mask_extensions = ["*.png"]

    x = []
    for ext in image_extensions:
        x.extend(sorted(glob(os.path.join(path, "images", ext))))

    y = []
    for ext in mask_extensions:
        y.extend(sorted(glob(os.path.join(path, "masks", ext))))
    return x, y

def shuffling(x, y):
  x, y = shuffle(x, y, random_state=42)
  return x, y

def split_dataset(path, split=0.1):
    """Загрузка изображений и масок"""
    X, Y = load_data(path)

    """Разбиение на тестовую и валидационную выборку"""
    split_size = int(len(X) * split)

    train_x, test_x = train_test_split(X, test_size=split_size, random_state=42)
    train_y, test_y = train_test_split(Y, test_size=split_size, random_state=42)

    return (train_x, train_y), (test_x, test_y)

def augment_data(images, masks, save_path, cfg: DictConfig, augment=True):
    def apply_augmentation(image, mask, aug):
        augmented = aug(image=image, mask=mask)
        return augmented["image"], augmented["mask"]

    def save_images(image, mask, name, index):
        temp_image_name = f"{name}_{index}.png"
        temp_mask_name = f"{name}_{index}.png"
        image_path = os.path.join(save_path, "images", temp_image_name)
        mask_path = os.path.join(save_path, "masks", temp_mask_name)
        cv2.imwrite(image_path, image)
        cv2.imwrite(mask_path, mask)

    def process_image(x, y, cfg: DictConfig):
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

    """Произведем аугментацию данных"""
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

        """Сохраняем данные"""
        index = 0
        for image, mask in zip(X, Y):
            image, mask = process_image(image, mask, cfg)
            save_images(image, mask, nameFile, index)
            index += 1
