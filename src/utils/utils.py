import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from omegaconf import DictConfig

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_dataset(path, split=0.1):
    train_x = sorted(glob(os.path.join(path, "train", "blurred_image", "*.jpg")))
    train_y = sorted(glob(os.path.join(path, "train", "mask", "*.png")))

    valid_x = sorted(glob(os.path.join(path, "validation", "P3M-500-NP", "original_image", "*.jpg")))
    valid_y = sorted(glob(os.path.join(path, "validation", "P3M-500-NP", "mask", "*.png")))

    return (train_x, train_y), (valid_x, valid_y)

def read_image(path, cfg: DictConfig):
    HEIGHT = cfg.resolution.HEIGHT
    WIDTH = cfg.resolution.WIDTH
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (WIDTH, HEIGHT))
    x = x / 255.0
    x = x.astype(np.float32)
    return x

def read_mask(path, cfg: DictConfig):
    path = path.decode()
    HEIGHT = cfg.resolution.HEIGHT
    WIDTH = cfg.resolution.WIDTH
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (WIDTH, HEIGHT))
    x = x / 255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x

def tf_parse(x, y, cfg: DictConfig):  # Передаем cfg как аргумент
    def _parse(x, y):
        x = read_image(x, cfg)  # Передаем cfg в read_image
        y = read_mask(y, cfg)  # Передаем cfg в read_mask
        return x, y

    HEIGHT = cfg.resolution.HEIGHT
    WIDTH = cfg.resolution.WIDTH
    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([HEIGHT, WIDTH, 3])
    y.set_shape([HEIGHT, WIDTH, 1])
    return x, y

def tf_dataset(X, Y, cfg: DictConfig, batch=2):  
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    ds = ds.map(lambda x, y: tf_parse(x, y, cfg)) 
    ds = ds.batch(batch).prefetch(10)
    return ds

def folder_path():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    return base_dir

def save_results(image, mask, y_pred, save_image_path):
    ## i - m - yp - yp*i
    HEIGHT = image.shape[0]
    line = np.ones((HEIGHT, 10, 3)) * 128

    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    #   mask = mask * 255

    y_pred = np.expand_dims(y_pred, axis=-1)    ## (512, 512, 1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)  ## (512, 512, 3)

    masked_image = image * y_pred
    y_pred = y_pred * 255

    cat_images = np.concatenate([image, line, mask, line, y_pred, line, masked_image], axis=1)
    cv2.imwrite(save_image_path, cat_images)