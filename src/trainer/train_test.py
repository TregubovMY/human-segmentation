import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Показывать только ошибки

import hydra
from omegaconf import DictConfig

from ..data.data_processing import load_data, shuffling
from ..utils.utils import tf_dataset, folder_path
from ..metrics.metrics import dice_loss, dice_coef, iou
from ..models.deepLabV3_plus import model_deepLabV3_plus
from ..models.u2_net import model_U2_Net, model_U2_Net_lite
from ..models.u_net import model_U_Net

import tensorflow as tf
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.optimizers import Adam
from keras.metrics import Recall, Precision
import numpy as np
import cv2

MODEL_FUNCTIONS = {
    "u_net": model_U_Net,
    "u2_net": model_U2_Net,
    "u2_net_lite": model_U2_Net_lite,
    "deepLabV3_plus": model_deepLabV3_plus,
}

LOSS_FUNCTION = {
    "u_net": dice_loss,
    "u2_net": "binary_crossentropy",
    "u2_net_lite": "binary_crossentropy",
    "deepLabV3_plus": dice_loss,
}

@hydra.main(config_path="../../config", config_name="main", version_base=None)
def main_train(cfg: DictConfig):
    """
    Обучает модель сегментации, используя конфигурацию Hydra.

    :param cfg: Конфигурация Hydra, содержащая параметры обучения.
    :type cfg: DictConfig
    """

    # Настройка зерна случайных чисел
    np.random.seed(42)
    tf.random.set_seed(42)

    # Загрузка параметров модели из конфигурации
    model_name = cfg.model.name
    model_function = MODEL_FUNCTIONS[model_name]
    batch_size = cfg.model.batch_size
    lr = cfg.model.lr
    num_epochs = cfg.model.num_epochs

    # Определение путей для сохранения модели и логов
    models_path = os.path.join(folder_path(), "models")
    csvs_path = os.path.join(folder_path(), "results", model_name)
    model_path = os.path.join(models_path, f"{model_name}.h5")
    csv_path = os.path.join(csvs_path, f"log_{model_name}.csv")

    # Загрузка данных для обучения и валидации
    base_dir = folder_path()
    train_path = os.path.join(base_dir, cfg.data.final, 'train')
    valid_path = os.path.join(base_dir, cfg.data.final, 'test')

    train_image_paths, train_mask_paths = load_data(train_path)
    train_image_paths, train_mask_paths = shuffling(train_image_paths, train_mask_paths)
    valid_image_paths, valid_mask_paths = load_data(valid_path)

    print(f"Train: {len(train_image_paths)} - {len(train_mask_paths)}")
    print(f"Valid: {len(valid_image_paths)} - {len(valid_mask_paths)}")

    # Создание TensorFlow Datasets для обучения и валидации
    train_dataset = tf_dataset(train_image_paths, train_mask_paths, cfg, batch=batch_size)
    valid_dataset = tf_dataset(valid_image_paths, valid_mask_paths, cfg, batch=batch_size)

    # Создание, компиляция и обучение модели
    model = model_function()
    model.compile(loss=LOSS_FUNCTION[model_name], 
                  optimizer=Adam(lr), 
                  metrics=[dice_coef, iou, Recall(), Precision()])

    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path, append=True),
        TensorBoard(),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False),
    ]

    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        callbacks=callbacks
    )

if __name__ == "__main__":
    main_train()
