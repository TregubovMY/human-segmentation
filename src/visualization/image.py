from .tools import *
import tensorflow as tf
from tqdm import tqdm
from glob import glob
from typing import List, Optional
from omegaconf import DictConfig

def apply_mask(image: np.ndarray, mask: np.ndarray, mode: str = "multiply") -> np.ndarray:
    """
    Применяет маску к изображению в соответствии с указанным режимом.

    :param image: Оригинальное изображение.
    :type image: np.ndarray
    :param mask: Маска для применения.
    :type mask: np.ndarray
    :param mode: Режим применения маски:
        - "multiply": умножает изображение на маску.
        - "concatenate": соединяет изображение и маску горизонтально.
        - "concatenate_multiplied": соединяет изображение, маску и результат умножения горизонтально.
        - По умолчанию: возвращает только маску.
    :type mode: str, optional
    :return: Изображение с примененной маской.
    :rtype: np.ndarray
    """
    if mode == "multiply":
        return masked_image(image, mask)
    elif mode == "concatenate":
        return np.concatenate([image, mask], axis=1)
    elif mode == "concatenate_multiplied":
        multiplied = masked_image(image, mask)
        mask = np.repeat(mask, 3, axis=2) 
        return np.concatenate([image, mask, multiplied], axis=1)
    else:
        return mask  # Default: Return only the mask

def present_results_on_models(dir_images: str, save_dir: str, models: List[tf.keras.Model], 
cfg: DictConfig, output_mode: str = "multiply", background_image: Optional[np.ndarray] = None) -> None:
    """
    Представляет результаты сегментации для нескольких моделей на изображениях из указанной директории.

    :param dir_images: Путь к директории с изображениями.
    :type dir_images: str
    :param save_dir: Путь к директории для сохранения результатов.
    :type save_dir: str
    :param models: Список моделей для сегментации.
    :type models: List[tf.keras.Model]
    :param cfg: Конфигурация, содержащая информацию о моделях и разрешении изображения.
    :type cfg: DictConfig
    :param output_mode: Режим применения маски (см. функцию `apply_mask`). 
                        По умолчанию "multiply".
    :type output_mode: str, optional
    """
    """ Настройка зерна случайных чисел """
    np.random.seed(42)
    tf.random.set_seed(42)

    for image_path in tqdm(glob(os.path.join(dir_images, "*"))):
        name = os.path.basename(image_path).split(".")[0]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        images = [image]
        for model in models:
            y = predict(image, model, cfg)
            masked_img = apply_mask(image, y, output_mode)
            masked_img = apply_background(image, y, background_image)
            images.append(masked_img)

        result_img = np.concatenate(images, axis=1)
        save_path = os.path.join(save_dir, f"{name}_all_models.png")
        save(result_img, save_path)


def show_pred_image(image: np.ndarray, model: tf.keras.Model) -> None:
    """
    Отображает предсказанную маску на изображении.

    :param image: Изображение для сегментации.
    :type image: np.ndarray
    :param model: Модель для сегментации.
    :type model: tf.keras.Model
    """
    if image is not None:
        y = predict(image, model)
        h, _, _ = image.shape
        masked_image = image * y
        line = np.ones((h, 10, 3)) * 128
        cat_images = np.concatenate([image, line, masked_image], axis=1)
        cv2.imshow("Prediction", cat_images)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Изображение не загружено.")
