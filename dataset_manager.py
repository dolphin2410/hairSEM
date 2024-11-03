#    hairSEM aims to quantify hair damage using a SEM image
#    Copyright (C) 2024 dolphin2410
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

from enum import Enum
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from ml_model import get_predicted_mask
from settings import LIMIT_DATASET_LOAD, TRAIN_TEST_RATIO

BLUISHNESS_THRESHOLD = 25  # TODO: This number is experimentally obtained

class ImageType(Enum):
  IMAGE = 1  # Raw SEM images
  MASK_IMAGE = 2  # Images with blue lines added on raw SEM images.
  MASK = 4  # Masks, extracted from mask_images, classified into 0s and 1s

  def from_bitmask(bitmask):
    types = []

    if bitmask & 0b001:
      types.append(ImageType.IMAGE)
    if bitmask & 0b010:
      types.append(ImageType.MASK_IMAGE)
    if bitmask & 0b100:
      types.append(ImageType.MASK)

    return types

def extract_masks(mask_image_path):
  """Function that extracts masks from mask_images"""
  
  mask_image = cv2.imread(mask_image_path)

  blues = mask_image[:, :, 0].copy().astype("int16")
  greens = mask_image[:, :, 1].copy().astype("int16")
  reds = mask_image[:, :, 2].copy().astype("int16")

  bluishness = np.vectorize(lambda x: 0 if x < 0 else x)(blues - greens - reds)
  maskified = np.vectorize(lambda x: 1 if x > BLUISHNESS_THRESHOLD else 0)(bluishness)
  
  return maskified

def create_masks_from_image(raw_images):
  images, masks = raw_images

  for mask in masks:
    cv2.imwrite(mask.replace("/segmentation-masks/", "/masks/"), extract_masks(mask))

def load_image_paths(dataset_directory, bitmask):
  """image paths from the given directory, returns only the types that were specified in the request bitmask"""
  
  requested_types = ImageType.from_bitmask(bitmask)
  paths = [[], [], []]

  files = os.listdir(dataset_directory + "/images")

  for file in files:
    image_path = os.path.join(dataset_directory + "/images/", file)
    mask_image_path = os.path.join(dataset_directory + "/segmentation-masks/", file)
    mask_path = os.path.join(dataset_directory + "/masks/", file)

    if ImageType.IMAGE in requested_types:
      paths[0].append(image_path)
    if ImageType.MASK_IMAGE in requested_types:
      paths[1].append(mask_image_path)
    if ImageType.MASK in requested_types:
      paths[2].append(mask_path)

  return np.array(list(filter(lambda x: len(x) != 0, paths)))

def load_images(image_paths):
  # TODO: why not make this random
  return np.array(list(map(lambda path: cv2.imread(path), image_paths[:LIMIT_DATASET_LOAD + 1])))

def normalize_image(images):
  return images.astype("float32") / 255.0

def split_data_with_ratio(data_list, ratio):
  test_max_index = int(len(data_list) * ratio)
  return data_list[:test_max_index + 1], data_list[test_max_index + 1:]

def load_dataset():
  images, masks = map(load_images, load_image_paths("sem_cropped_images", ImageType.IMAGE.value + ImageType.MASK.value))
  test_images, train_images = split_data_with_ratio(normalize_image(images), TRAIN_TEST_RATIO)
  test_masks, train_masks = split_data_with_ratio(masks, TRAIN_TEST_RATIO)
  return test_images, train_images, test_masks, train_masks

def get_original_samples(image: np.ndarray):
  list_masks = []
  x_shape, y_shape = image.shape
  for x in range(0, x_shape, 128):
    for y in range(0, y_shape, 128):
      cropped_image = image[y:y+128, x:x+128]
      predicted_mask = get_predicted_mask(cropped_image)
      list_masks.append(predicted_mask)
  
  return predicted_mask


if __name__ == "__main__":
  # For debugging purposes
  test_images, train_images, test_masks, train_masks = load_dataset()