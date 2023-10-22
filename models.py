import os
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.layers import GlobalAveragePooling2D, Dense
from keras.applications import ResNet50
from keras.callbacks import ReduceLROnPlateau

def load_one_image(path,label):
    """
    load one image 
    """
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image)
    return image,label

def build_model(num_out):
    """
    This is architecture for model
    """
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    global_average_layer = GlobalAveragePooling2D()(base_model.get_layer("conv5_block1_out").output)
    dense_layer = Dense(128, activation='relu')(global_average_layer)
    output_layer = Dense(num_out, activation='softmax')(dense_layer)
    for layer in base_model.layers:
        layer.trainable = False
    model = tf.keras.models.Model(inputs=base_model.input, outputs=output_layer)
    return model


def save_plot(out_dir,history):
    """
    This function is to save the model history to runtime file.
    """
    fig,ax=plt.subplots(1,2,figsize=(10,5))
    ax[0].plot(history.history["loss"])
    ax[0].plot(history.history["val_loss"])
    ax[0].legend(["loss","val_loss"],loc="best")
    ax[0].set_title("Model loss")
    ax[0].set_xlabel("epochs")
    ax[0].set_ylabel("loss")

    ax[1].plot(history.history["accuracy"])
    ax[1].plot(history.history["val_accuracy"])
    ax[1].legend(["accuracy","val_accuracy"],loc="best")
    ax[1].set_title("Model accuracy")
    ax[1].set_xlabel("epochs")
    ax[1].set_ylabel("accuracy")

    plt.savefig(os.path.join(out_dir,"history.png"))

reduce_lr_callback = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.25,
    patience=3,
    min_lr=1e-6,
    verbose=1
)