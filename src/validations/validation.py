import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import csv

import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

from ..metrics.metrics import *
from ..utils.utils import save_results, read_image
from ..models.deepLabV3_plus import model_deepLabV3_plus
from ..models.u2_net import model_U2_Net, model_U2_Net_lite
from ..models.u_net import model_U_Net
from ..data.data_processing import load_data

MODEL_FUNCTIONS = {
    "u_net": model_U_Net,
    "u2_net": model_U2_Net,
    "u2_net_lite": model_U2_Net_lite,
    "deepLabV3_plus": model_deepLabV3_plus,
}

@hydra.main(config_path="../../config", config_name="main", version_base=None)
def val(cfg: DictConfig):
    """
    Выполняет валидацию модели сегментации на тестовом наборе данных.

    :param cfg: Конфигурация Hydra, содержащая параметры модели и пути к данным.
    :type cfg: DictConfig
    """
    height = cfg.resolution.HEIGHT
    width = cfg.resolution.WIDTH

    """ Настройка зерна случайных чисел """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Загрузка модели """
    model_name = cfg.model.name
    model_function = MODEL_FUNCTIONS[model_name]
    model = model_function()

    """ Загрузка набора данных """
    valid_path = cfg.data.validation
    test_image_paths, test_mask_paths = load_data(valid_path)
    print(f"Test: {len(test_image_paths)} - {len(test_mask_paths)}")

    """ Оценка и предсказание """
    scores = []
    with open(f"files/score_{model_name}.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image", "Accuracy", "F1", "Jaccard", "Recall", "Precision"])
        for image_path, mask_path in tqdm(zip(test_image_paths, test_mask_paths), total=len(test_image_paths)):
            """ Извлечение имени """
            name = os.path.basename(image_path).split('.')[0]

            """ Чтение изображения """
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            original_height, original_width, _ = image.shape
            image = cv2.resize(image, (width, height))
            image = image / 255.0
            image = np.expand_dims(image, axis=0)

            """ Чтение маски """
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            """ Предсказание """
            y_pred = model.predict(image, verbose=0)

            # Проверка, является ли модель U2-Net
            if isinstance(y_pred, list):
                y_pred = y_pred[0][0]
            else:
                y_pred = y_pred[0]
            y_pred = cv2.resize(y_pred, (original_width, original_height))
            y_pred = y_pred > 0.5

            """ Сохранение предсказания """
            save_image_path = f"results/{model_name}/images/{name}.png"
            save_results(image, mask, y_pred, save_image_path)

            """ Выравнивание массива """
            mask = mask.flatten()
            y_pred = y_pred.flatten()

            y_pred = y_pred.astype(np.int32)
            mask = mask > 0.5

            """ Вычисление значений метрик """
            acc_value = accuracy_score(mask, y_pred)
            f1_value = f1_score(mask, y_pred, labels=[0, 1], average="binary")
            jac_value = jaccard_score(mask, y_pred, labels=[0, 1], average="binary")
            recall_value = recall_score(mask, y_pred, labels=[0, 1], average="binary")
            precision_value = precision_score(mask, y_pred, labels=[0, 1], average="binary")
            scores.append([acc_value, f1_value, jac_value, recall_value, precision_value])
            writer.writerow([name, acc_value, f1_value, jac_value, recall_value, precision_value])

    """ Значения метрик """
    score = [s for s in scores]
    score = np.mean(score, axis=0)
    print(f"Accuracy: {score[0]:0.5f}")
    print(f"F1: {score[1]:0.5f}")
    print(f"Jaccard: {score[2]:0.5f}")
    print(f"Recall: {score[3]:0.5f}")
    print(f"Precision: {score[4]:0.5f}")
    writer.writerow(["Average", score[0], score[1], score[2], score[3], score[4]])

if __name__ == "__main__":
    val()
