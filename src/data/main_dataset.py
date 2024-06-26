import os
from .data_processing import *
import numpy as np
import hydra
from ..utils.utils import folder_path, create_dir 

@hydra.main(config_path="./../../config", config_name="main", version_base=None)
def main_dataset(cfg: DictConfig):
    # Получаем путь к верхнему каталогу
    base_dir = folder_path()
    data_raw = os.path.join(base_dir, cfg.data.raw)
    data_final = os.path.join(base_dir, cfg.data.final)

    """Seeding"""
    np.random.seed(42)

    """Подгружаем данные"""
    (train_x, train_y), (test_x, test_y) = split_dataset(data_raw)

    if not os.path.isdir(data_final + "train\images"):
        """Создаем выходные директории"""
        create_dir(data_final + "train\images")
        create_dir(data_final + "train\masks")
        create_dir(data_final + "test\images")
        create_dir(data_final + "test\masks")

    # Флаг аугментации из конфигурации
    augment = cfg.get("augment", False)

    if not os.listdir(data_final + "train\images"):
        """Data augmentation"""
        augment_data(train_x, train_y, data_final + "train", cfg, augment=augment)
        augment_data(test_x, test_y, data_final + "test", cfg, augment=False)  # Тестовые данные не аугментируем


if __name__ == "__main__":
    main_dataset()
