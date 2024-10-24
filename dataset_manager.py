import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from settings import LIMIT_DATASET_LOAD

BLUISHNESS_THRESHOLD = 25  # TODO: This number is experimentally obtained

def extract_masks(file_path):
  print(file_path)
  mask_image = cv2.imread(file_path)
  blues = mask_image[:, :, 0].copy().astype("int16")
  greens = mask_image[:, :, 1].copy().astype("int16")
  reds = mask_image[:, :, 2].copy().astype("int16")

  bluishness = np.vectorize(lambda x: 0 if x < 0 else x)((blues - greens - reds))
  maskified = np.vectorize(lambda x: 1 if x > BLUISHNESS_THRESHOLD else 0)(bluishness)
  
  return maskified

def load_directory(root_dir):
  image_paths = []
  mask_image_paths = []
  mask_paths = []

  files = os.listdir(root_dir + "/images")
  for file in files:
    image_path = os.path.join(root_dir + "/images/", file)
    mask_image_path = os.path.join(root_dir + "/segmentation-masks/", file)
    mask_path = os.path.join(root_dir + "/masks/", file)
    
    image_paths.append(image_path)
    mask_image_paths.append(mask_image_path)
    mask_paths.append(mask_path)

  return np.array(image_paths), np.array(mask_paths)

def load_images(image_paths, limit):
  images = []
  counter = 0
  for path in image_paths:
    if counter == limit:
      break
    counter += 1
    images.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
  return np.array(images)

def split_loaded_images(data_list, ratio):
  test_max_index = int(len(data_list) * ratio) + 1
  return data_list[:test_max_index], data_list[test_max_index:]

def create_masks_from_image(raw_images):
  images, masks = raw_images

  for mask in masks:
    cv2.imwrite(mask.replace("/segmentation-masks/", "/masks/"), extract_masks(mask))

def normalize_image(images):
  return images.astype("float32") / 255.0

def train_test_split():
  images, masks = map(lambda image_paths: load_images(image_paths, LIMIT_DATASET_LOAD), load_directory("sem_cropped_images"))
  test_images, train_images = split_loaded_images(normalize_image(images), 0.3)
  test_masks, train_masks = split_loaded_images(masks, 0.3)