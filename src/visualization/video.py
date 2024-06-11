import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
from .tools import predict
from tqdm import tqdm
from typing import List
from omegaconf import DictConfig
import tensorflow as tf

def process_videos_in_folder(input_folder: str, output_folder: str, models: List[tf.keras.Model], cfg: DictConfig, fps: int = 30) -> None:
    """
    Обрабатывает все видеофайлы в указанной папке.

    :param input_folder: Путь к папке с видеофайлами.
    :type input_folder: str
    :param output_folder: Путь к папке для сохранения обработанных видеофайлов.
    :type output_folder: str
    :param models: Список моделей для сегментации.
    :type models: List[Model]
    :param cfg: Конфигурация для моделей.
    :type cfg: Dict
    :param fps: Частота кадров в секунду для выходного видео. По умолчанию равна 30.
    :type fps: int
    :return: None
    """

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.mp4', '.avi', '.mkv')):
            video_path = os.path.join(input_folder, filename)
            processing_video(video_path, output_folder, models, cfg, fps)

def processing_video(video_path: str, output_dir: str, models:  List[tf.keras.Model], cfg: DictConfig, fps: int = 30) -> None:
    """
    Обрабатывает видео, применяя модели сегментации и добавляя подписи.

    :param video_path: Путь к видеофайлу.
    :type video_path: str
    :param output_dir: Путь к папке для сохранения обработанного видеофайла.
    :type output_dir: str
    :param models: Список моделей для сегментации.
    :type models: List[Model]
    :param cfg: Конфигурация для моделей.
    :type cfg: Dict
    :param fps: Частота кадров в секунду для выходного видео. По умолчанию равна 30.
    :type fps: int
    :return: None
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    target_width = 512
    target_height = 512

    num_models = len(models)
    output_width = target_width * (num_models + 1) if num_models != 3 else target_width * 2
    output_height = target_height * 2 if num_models == 3 else target_height

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    name = os.path.basename(video_path).split('.')[0]
    output_path = os.path.join(output_dir, f"{name}.avi")
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height), True)

    model_names = ["model_deeplabV3_plus", "model_u_net", "model_u2_net"]

    with tqdm(total=total_frames, desc=f"Обработка видео: {name}") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (target_width, target_height))
            masks = [predict(frame, model, cfg) for model in models]
            resized_masks = [np.expand_dims(cv2.resize(mask, (target_width, target_height)), axis=2) for mask in masks]

            # Создаем список сегментированных кадров с подписями
            segmented_frames = []
            for i, mask in enumerate(resized_masks):
                segmented_frame = frame * mask
                cv2.putText(segmented_frame, model_names[i], (100, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                segmented_frames.append(segmented_frame)

            # Объединяем кадры в зависимости от количества моделей
            if num_models == 3:
                top = np.concatenate((frame, segmented_frames[0]), axis=1)
                bottom = np.concatenate((segmented_frames[1], segmented_frames[2]), axis=1)
                combine_frame = np.concatenate((top, bottom), axis=0)
            else:
                combine_frame = frame
                for segmented_frame in segmented_frames:
                    combine_frame = np.concatenate((combine_frame, segmented_frame), axis=1)

            combine_frame = combine_frame.astype(np.uint8)
            out.write(combine_frame)
            pbar.update(1)

    cap.release()
    out.release()
