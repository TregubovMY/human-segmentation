import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from keras.utils import CustomObjectScope
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

from src.data.data_processing import load_data
from src.utils.utils import create_dir, folder_path, save_results

from src.models.deeplabv3plus import model_deepLabV3_plus
from src.models.u2_net import model_U2_Net
from src.models.u_net import model_U_Net

import hydra
from omegaconf import DictConfig

MODEL_FUNCTIONS = {
    "u_net": model_U_Net,
    "u2_net": model_U2_Net,
    "deepLabV3_plus": model_deepLabV3_plus,
}

def load_model(model_name):
    """Загружает модель по имени."""
    return MODEL_FUNCTIONS[model_name]()

def prepare_data(dataset_path):
    """Загружает и подготавливает набор данных."""
    test_x, test_y = load_data(dataset_path)
    print(f"Test: {len(test_x)} - {len(test_y)}")
    return test_x, test_y

def predict_and_evaluate(model, test_x, test_y, result_path):
    """Выполняет предсказание и оценку модели."""
    scores = []
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        name = os.path.basename(x).split('.')[0]

        image = cv2.imread(x, cv2.IMREAD_COLOR)
        h, w, _ = image.shape
        image_resized = cv2.resize(image, (512, 512))

        x = image_resized / 255.0
        x = np.expand_dims(x, axis=0)
        
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)

        y_pred = model.predict(x, verbose=0)[0]
        y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = cv2.resize(y_pred, (w, h), interpolation=cv2.INTER_NEAREST)
        y_pred = (y_pred > 0.5).astype(np.int32)

        save_image_path = os.path.join(result_path, "images", f"{name}.png")
        create_dir(os.path.join(result_path, "images"))
        save_results(image, mask, y_pred, save_image_path)

        mask = mask > 0.5
        scores.append([
            name,
            accuracy_score(mask.flatten(), y_pred.flatten()),
            f1_score(mask.flatten(), y_pred.flatten(), labels=[0, 1], average="binary"),
            jaccard_score(mask.flatten(), y_pred.flatten(), labels=[0, 1], average="binary"),
            recall_score(mask.flatten(), y_pred.flatten(), labels=[0, 1], average="binary"),
            precision_score(mask.flatten(), y_pred.flatten(), labels=[0, 1], average="binary")
        ])

    return scores

def save_metrics(scores, result_path, model_name):
    """Сохраняет метрики в CSV файл."""
    score_values = np.mean(np.array(scores)[:, 1:].astype(float), axis=0)
    print(f"Accuracy: {score_values[0]:0.5f}")
    print(f"F1: {score_values[1]:0.5f}")
    print(f"Jaccard: {score_values[2]:0.5f}")
    print(f"Recall: {score_values[3]:0.5f}")
    print(f"Precision: {score_values[4]:0.5f}")

    df = pd.DataFrame(scores, columns=["Image", "Accuracy", "F1", "Jaccard", "Recall", "Precision"])
    df.to_csv(os.path.join(result_path, f"score_{model_name}.csv"))

@hydra.main(config_path="../../config", config_name="main", version_base=None)
def main(cfg: DictConfig):
    np.random.seed(42)
    tf.random.set_seed(42)

    model_name = cfg.model.name
    model = load_model(model_name)

    base_dir = folder_path()
    result_path = os.path.join(base_dir, "results", model_name)

    dataset_path = os.path.join(base_dir, cfg.data.validation)
    test_x, test_y = prepare_data(dataset_path)

    scores = predict_and_evaluate(model, test_x, test_y, result_path)
    save_metrics(scores, result_path, model_name)

if __name__ == "__main__":
    main()