from functools import partial
from tqdm import tqdm
from models.uNet import U_Net
from evaluation.metrics import *
from data.make_dataset1 import *
import requests
import numpy as np
import cv2
from keras.utils import CustomObjectScope
from glob import glob
import tensorflow as tf
from keras.utils import CustomObjectScope