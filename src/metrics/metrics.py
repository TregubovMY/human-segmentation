import numpy as np
import tensorflow as tf

def iou(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Вычисляет коэффициент Jaccard (IoU) между двумя тензорами.

    Args:
        y_true: Тензор истинных значений.
        y_pred: Тензор предсказанных значений.

    Returns:
        tf.Tensor: Коэффициент IoU.
    """
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

smooth = 1e-15
def dice_coef(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Вычисляет коэффициент Дайса (Dice coefficient) между двумя тензорами.

    Args:
        y_true: Тензор истинных значений.
        y_pred: Тензор предсказанных значений.

    Returns:
        tf.Tensor: Коэффициент Дайса.
    """
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Вычисляет функцию потерь на основе коэффициента Дайса (Dice loss).

    Args:
        y_true: Тензор истинных значений.
        y_pred: Тензор предсказанных значений.

    Returns:
        tf.Tensor: Значение функции потерь.
    """
    return 1.0 - dice_coef(y_true, y_pred)