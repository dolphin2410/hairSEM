import numpy as np

from settings import BOX_SIZE


MASK_THRESHOLD = 100

def flatten_predicted_mask(image: np.ndarray):
    x_size, y_size, _ = image.shape

    for x in range(0, x_size - BOX_SIZE + 1, BOX_SIZE):
        for y in range(0, y_size - BOX_SIZE + 1, BOX_SIZE):
            min = image[x: x+BOX_SIZE, y: y+BOX_SIZE, :].min()
            min = 0 if min < MASK_THRESHOLD else 255
            image[x: x+BOX_SIZE, y: y+BOX_SIZE] = np.array([[[min, min, min]] * BOX_SIZE] * BOX_SIZE)

    return image