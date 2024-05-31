import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Показывать только ошибки

import hydra
from omegaconf import DictConfig

from src.data.data_processing import load_data, shuffling
from src.utils.utils import tf_dataset, folder_path, create_dir
from src.metrics.metrics import dice_loss, dice_coef, iou
from src.models.deeplabv3plus import model_deepLabV3_plus
from src.models.u2_net import model_U2_Net
from src.models.u_net import model_U_Net

import tensorflow as tf
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.optimizers import Adam
from keras.metrics import Recall, Precision
import numpy as np

MODEL_FUNCTIONS = {
    "u_net": model_U_Net,
    "u2_net": model_U2_Net,
    "deepLabV3_plus": model_deepLabV3_plus,
}

@hydra.main(config_path="../../config", config_name="main", version_base=None)
def main_train(cfg: DictConfig):
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Хранение итоговых файлов """
    models_path = os.path.join(folder_path(), "models")
    csvs_path = os.path.join(folder_path(), "results")

    """ Параметры модели """
    model_name = cfg.model.name
    model_function = MODEL_FUNCTIONS[model_name]
    batch_size = cfg.model.batch_size
    lr = cfg.model.lr
    num_epochs = cfg.model.num_epochs

    model_path = os.path.join(models_path, f"model_{model_name}.h5")
    csv_path = os.path.join(csvs_path, f"data_{model_name}.csv")

    """ Набор данных """
    base_dir = folder_path()
    train_path = os.path.join(base_dir, cfg.data.final, 'train')
    valid_path = os.path.join(base_dir, cfg.data.final, 'test')

    train_x, train_y = load_data(train_path)
    train_x, train_y = shuffling(train_x, train_y)
    valid_x, valid_y = load_data(valid_path)

    train_dataset = tf_dataset(train_x, train_y, cfg, batch=batch_size)  # (8,512,512,3)
    valid_dataset = tf_dataset(valid_x, valid_y, cfg, batch=batch_size)  # (8,512,512,1)

    """ Модель """
    model = model_function()
    model.compile(loss=dice_loss, optimizer=Adam(lr), metrics=[dice_coef, iou, Recall(), Precision()])

    """"""
    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path, append=True),
        TensorBoard(),
        EarlyStopping(monitor='val_loss', patience=11, restore_best_weights=False),
    ]

    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        callbacks=callbacks
    )


if __name__ == "__main__":
    main_train()