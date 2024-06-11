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
    HEIGHT = cfg.resolution.HEIGHT
    WIDTH = cfg.resolution.WIDTH
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Loading model """
    model_name = cfg.model.name
    model_function = MODEL_FUNCTIONS[model_name]
    model = model_function()

    """ Load the dataset """
    valid_path = cfg.data.validation
    test_x, test_y = load_data(valid_path)
    print(f"Test: {len(test_x)} - {len(test_y)}")

    """ Evaluation and Prediction """
    SCORE = []
    with open(f"files/score_n2_net_lite.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image", "Accuracy", "F1", "Jaccard", "Recall", "Precision"])
        for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
            """ Extract the name """
            name = os.path.basename(x).split('.')[0]

            """ Reading the image """
            image = cv2.imread(x, cv2.IMREAD_COLOR)
            h, w, _ = image.shape
            x = cv2.resize(image, (HEIGHT, WIDTH))

            x = x/255.0
            x = np.expand_dims(x, axis=0)

            """ Reading the mask """
            mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)

            """ Prediction """
            y_pred = model.predict(x, verbose=0)

            # Проверка, является ли модель U2-Net
            if isinstance(y, list):
                y_pred = y_pred[0][0]
            else:
                y_pred = y_pred[0]
            y_pred = cv2.resize(y_pred, (w, h))
            y_pred = y_pred > 0.5

            """ Saving the prediction """
            save_image_path = f"results/{model_name}/images/{name}.png"
            save_results(image, mask, y_pred, save_image_path)

            """ Flatten the array """
            mask = mask.flatten()
            y_pred = y_pred.flatten()

            y_pred = y_pred.astype(np.int32)
            mask = mask > 0.5
            """ Calculating the metrics values """
            acc_value = accuracy_score(mask, y_pred)
            f1_value = f1_score(mask, y_pred, labels=[0, 1], average="binary")
            jac_value = jaccard_score(mask, y_pred, labels=[0, 1], average="binary")
            recall_value = recall_score(mask, y_pred, labels=[0, 1], average="binary")
            precision_value = precision_score(mask, y_pred, labels=[0, 1], average="binary")
            SCORE.append([acc_value, f1_value, jac_value, recall_value, precision_value])
            writer.writerow([name, acc_value, f1_value, jac_value, recall_value, precision_value])

    """ Metrics values """
    score = [s for s in SCORE]
    score = np.mean(score, axis=0)
    print(f"Accuracy: {score[0]:0.5f}")
    print(f"F1: {score[1]:0.5f}")
    print(f"Jaccard: {score[2]:0.5f}")
    print(f"Recall: {score[3]:0.5f}")
    print(f"Precision: {score[4]:0.5f}")
    writer.writerow(["Average", score[0], score[1], score[2], score[3], score[4]])
