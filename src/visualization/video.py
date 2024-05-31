import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
from tools import predict

def extract_path_info(file_path):
  directory = os.path.dirname(file_path)
  file_name = os.path.basename(file_path)
  name, extension = os.path.splitext(file_name)
  return directory, name, extension

def processing_video(video_path, model, fps = 30):
  """ Reading frames """
  cap = cv2.VideoCapture(video_path)
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  fourcc = cv2.VideoWriter_fourcc(*'MJPG')
  dir, name, _ = extract_path_info(video_path)
  out = cv2.VideoWriter(f"{dir}/{name}_mask.mp4", fourcc, fps, (width, height), True)

  idx = 0
  while cap.isOpened():
    ret, frame = cap.read()
    if ret == False:
      break
    mask = predict(frame, model)

    combine_frame = frame * mask
    combine_frame = combine_frame.astype(np.uint8)

    out.write(combine_frame)

  cap.release()
  out.release()