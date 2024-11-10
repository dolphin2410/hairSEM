from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

from settings import BOX_SIZE
from scipy.ndimage import label

MASK_THRESHOLD = 130

def flatten_predicted_mask(image: np.ndarray):
    """2x2 박스로 전체 이미지를 나누어 flattening 작업을 수행한다"""

    x_size, y_size, _ = image.shape

    for x in range(0, x_size - BOX_SIZE + 1, BOX_SIZE):
        for y in range(0, y_size - BOX_SIZE + 1, BOX_SIZE):
            # 각 상자의 최솟값으로 픽셀을 치환
            min = image[x: x+BOX_SIZE, y: y+BOX_SIZE, :].min()
            min = 0 if min < MASK_THRESHOLD else 255
            if x == 0 or y == 0:
                min = 0
            image[x: x+BOX_SIZE, y: y+BOX_SIZE] = np.array([[[min, min, min]] * BOX_SIZE] * BOX_SIZE)

    return image

def coerce_image(image: np.ndarray):
   """최댓값과 최솟값을 기준으로 픽셀을 0~255로 조정"""

   max, min = image.max(), image.min()

   coerced_image = (image - min) * 255 / (max - min)

   return np.vectorize(lambda x: int(x))(coerced_image)

def chunkify(image: np.ndarray):
    """마스크에서 군집을 추출해낸다"""

    x_size, y_size, _ = image.shape

    # 군집 레이블링
    
    marker, ncc = label(image)
    marked_pixels = [[] for i in range(ncc + 1)]

    for y in range(y_size):
        for x in range(x_size):
            # 검은색 픽셀이라면 군집에 포함시키지 않는다
            if image[y, x, 0] == 0:
                continue
            
            r, g, b = marker[y, x]
            if r != g or g != b:
                raise ValueError("invalid marker")
            
            # 군집 별로 픽셀의 좌표를 대입한다
            marked_pixels[r].append((x, y))

    return list(filter(lambda x: len(x) != 0, marked_pixels))

@tf.function
def cost_function(gradient, X, Y, bias, N):
    """선형회귀 - MSE 손실함수 정의"""

    pred = tf.add(tf.multiply(X, gradient), bias)
    offset = tf.subtract(pred, Y)
    loss = tf.reduce_sum(tf.multiply(offset, offset))
    
    return tf.divide(loss, N)

def mask_linear_regression(gradient, pixels):
    """마스크의 군집에 대해 선형회귀 진행 및 최소 MSE 반환"""

    # 텐서 정의

    bias = tf.Variable(0.0, name = "b")
    gradient_tensor = tf.constant(gradient)
    N = tf.constant(float(len(pixels)))
    x = tf.constant(list(map(lambda x: float(x[0]), pixels)))
    y = tf.constant(list(map(lambda x: float(x[1]), pixels)))

    # 학습 - 경사하강

    @tf.function
    def train(learning_rate=0.03):
        with tf.GradientTape() as t:
            current_loss = cost_function(gradient_tensor, x, y, bias, N)
            dBias = t.gradient(current_loss, [bias])[0]

        bias.assign_sub(learning_rate * dBias)
        return current_loss

    # 300회 반복 학습
    for epoch in range(100):
        loss = train()

    loss = cost_function(gradient, x, y, bias, len(pixels))
    
    return float(loss.numpy() * N.numpy())