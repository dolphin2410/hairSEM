"""
Portions of this code are licensed under the Apache License, Version 2.0. 
See the LICENSE-APACHE file for details.
"""

import uuid
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from IPython.display import clear_output
from tensorflow_examples.tensorflow_examples.models.pix2pix import pix2pix
from dataset_manager import load_dataset

test_images, train_images, test_masks, train_masks = load_dataset()

sample_image, sample_mask = train_images[2], train_masks[2]

print(test_images.shape)
  
def coerce_image(image: np.ndarray):
   max, min = image.max(), image.min()

   coerced_image = (image - min) * 255 / (max - min)

   return np.vectorize(lambda x: int(x))(coerced_image)

class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        
        global model

        title = ['Input Image', 'True Mask', 'Predicted Mask']
        display_list = [sample_image, sample_mask * 255, coerce_image(np.array(model.predict(np.array([sample_image]))[0]))]

        if epoch % 20 == 0:
            plt.figure(figsize=(15, 15))
            for i in range(3):
                plt.subplot(1, 3, i+1)
                plt.title(title[i])
                plt.imshow(display_list[i])
                plt.colorbar()
                plt.axis('off')
            plt.show()

        model.save(f'models/epoch{epoch}-{uuid.uuid4()}.keras')

        print(f'\nSample Prediction after epoch {epoch + 1}\n')

def unet_model(output_channels = 3): # output_channels : number of classes
    base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
    ]

    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

    down_stack.trainable = False

    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),   # 32x32 -> 64x64
    ]

    inputs = tf.keras.layers.Input(shape=[128, 128, 3])

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides=2,
        padding='same')  #64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

global model
model = unet_model()
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model_history = model.fit(train_images, train_masks, epochs=200, batch_size=64, validation_data=(test_images, test_masks), callbacks=[DisplayCallback()])
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

plt.figure()
plt.plot(model_history.epoch, loss, 'r', label='Training loss')
plt.plot(model_history.epoch, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()