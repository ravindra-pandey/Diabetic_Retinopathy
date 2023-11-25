import os
import numpy as np
import pandas as pd
import tensorflow as tf

import models as md

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print("gpu", gpu, end="/n")
    except RuntimeError as e:
        print(e)

nonretina_dataset = tf.keras.datasets.cifar100


def prepare_dataset():
    """
    This function is responsible for the prepration of the dataset
    """
    (x_train, y_train), (x_val, y_val) = nonretina_dataset.load_data()
    nonretina_images = np.append(x_train, x_val, axis=0)
    nonretina_label = np.zeros(nonretina_images.shape[0])
    image_names = [
        f"{root}/{fl}" for root, dirs, files in os.walk("augmented_dataset") for fl in files
    ]

    np.random.shuffle(image_names)

    retina_images = np.array(
        [
            tf.image.resize(md.load_one_image(img_name, "_")[0], (32, 32))
            for img_name in image_names[:60000]
        ]
    )
    retina_label = np.ones(retina_images.shape[0])
    # print(nonretina_images.shape,retina_images.shape)
    X = np.append(nonretina_images, retina_images, axis=0)
    y = np.append(nonretina_label, retina_label, axis=0)
    return X, y


X, y = prepare_dataset()

print(set(y))
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer="adam",loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=["accuracy"])
print(model.summary())

history=model.fit(X,y,batch_size=32,epochs=10,validation_split=0.2,callbacks=[md.reduce_lr_callback])
model.save("models/retina_detection.h5")

pd.DataFrame(history.history).to_csv("retina_detection_model_history.csv")
