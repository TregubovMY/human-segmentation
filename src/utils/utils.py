import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import tensorflow as tf
from omegaconf import DictConfig

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_image(path, cfg: DictConfig):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    if cfg.model.name in ["u2_net", "u2_net_lite"]: # траблы
        x = cv2.resize(x, (cfg.resolution.WIDTH, cfg.resolution.HEIGHT))

    x = x / 255.0
    x = x.astype(np.float32)
    return x

def read_mask(path, cfg: DictConfig):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if cfg.model.name in ["u2_net", "u2_net_lite"]:
        x = cv2.resize(x, (cfg.resolution.WIDTH, cfg.resolution.HEIGHT))
        x = x / 255.0

    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x

def tf_parse(x, y, cfg: DictConfig):
    def _parse(x, y):
        x = read_image(x, cfg)
        y = read_mask(y, cfg)
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
    # mask = mask * 255 # для некоторых решений это не нужно

    y_pred = np.expand_dims(y_pred, axis=-1)    ## (512, 512, 1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)  ## (512, 512, 3)

    masked_image = image * y_pred
    y_pred = y_pred * 255

    cat_images = np.concatenate([image, line, mask, line, y_pred, line, masked_image], axis=1)
    cv2.imwrite(save_image_path, cat_images)