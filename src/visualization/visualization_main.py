import os
import argparse
from .image import present_results_on_models
from .video import processing_video, process_videos_in_folder
from ..models.deepLabV3_plus import model_deepLabV3_plus
from ..models.u2_net import model_U2_Net, model_U2_Net_lite
from ..models.u_net import model_U_Net
import hydra
from omegaconf import DictConfig

# Словарь для связывания имен моделей с функциями загрузки
MODEL_LOADERS = {
    "model_u_net": model_U_Net,
    "model_deeplabv3_plus": model_deepLabV3_plus,
    "model_u2_net": model_U2_Net,
    "model_u2_net_lite": model_U2_Net_lite,
}

def main(cfg: DictConfig) -> None:
    """
    Главная функция, которая выполняет сегментацию изображений или видео.

    :param cfg: Конфигурация для сегментации.
    :type cfg: DictConfig
    :return: None
    """
    
    # Разбор аргументов командной строки для переопределения конфигурации
    parser = argparse.ArgumentParser(
        description="Сегментация изображений и видео",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["image", "video"], help="Режим: 'image' или 'video'."
    )
    parser.add_argument("--input", help="Путь к входному изображению/видео.")
    parser.add_argument("--output", help="Путь для сохранения.")
    parser.add_argument(
        "--models", help="Список моделей (через пробел).", nargs="+"
    )
    parser.add_argument(
        "--fps", type=int, help="Количество кадров в секунду для видео."
    )
    args = parser.parse_args()

    # Переопределение параметров конфигурации из аргументов командной строки
    if args.mode:
        cfg.visualization.mode = args.mode
    if args.input:
        cfg.visualization.input = args.input
    if args.output:
        cfg.visualization.output = args.output
    if args.models:
        cfg.visualization.models = args.models
    if args.fps:
        cfg.visualization.fps = args.fps

    # Получение параметров из конфигурации Hydra
    mode = cfg.visualization.mode
    input_path = cfg.visualization.input
    output_path = cfg.visualization.output
    model_names = cfg.visualization.models
    fps = cfg.visualization.fps

    # Загрузка моделей
    loaded_models = []
    for model_name in model_names:
        model_name = model_name.lower()
        if model_name not in MODEL_LOADERS:
            raise ValueError(f"Неизвестная модель: {model_name}")
        loaded_models.append(MODEL_LOADERS[model_name]())

    if not loaded_models:
        raise ValueError("Необходимо указать хотя бы одну модель.")

    # Создание выходного каталога
    os.makedirs(output_path, exist_ok=True)

    # Выполнение сегментации
    if mode == "image":
        present_results_on_models(
            input_path,
            output_path,
            loaded_models,
            cfg,
            output_mode="multiply",
        )
    elif mode == "video":
      if os.path.isfile(input_path):
          print()
          processing_video(input_path, output_path, loaded_models, cfg, fps=fps)
      elif os.path.isdir(input_path):
          process_videos_in_folder(input_path, output_path, loaded_models, cfg, fps=fps)
      else:
          raise ValueError("Укажите правильный путь к видеофайлу или папке.")

if __name__ == "__main__":
    main()
