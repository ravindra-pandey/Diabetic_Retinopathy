import os,datetime
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
plt.style.use('seaborn-v0_8-dark')

import tensorflow as tf
from keras.layers import GlobalAveragePooling2D, Dense
from keras.applications import ResNet50
from keras.callbacks import ReduceLROnPlateau
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print("gpu",gpu,end="/n")
    except RuntimeError as e:
        print(e)



image_names = [os.path.join(root, fl) for root, dirs, files in os.walk("augmented_dataset") for fl in files]
np.random.shuffle(image_names)
labels = [0  if i.split("/")[-2] == "normal" else 1  for i in image_names]

X_train,X_test,y_train,y_test=train_test_split(image_names,labels,test_size=0.2,random_state=42)


def load_one_image(path,label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image)
    return image,label

AUTOTUNE = tf.data.AUTOTUNE
train_dataset=tf.data.Dataset.from_tensor_slices((X_train,y_train)).map(load_one_image,num_parallel_calls=AUTOTUNE).cache().batch(batch_size=16).prefetch(AUTOTUNE)
test_dataset=tf.data.Dataset.from_tensor_slices((X_test,y_test)).map(load_one_image,num_parallel_calls=AUTOTUNE).cache().batch(batch_size=16).prefetch(AUTOTUNE)

def build_model(num_out):
  base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
  global_average_layer = GlobalAveragePooling2D()(base_model.get_layer("conv5_block2_out").output)
  dense_layer = Dense(128, activation='relu')(global_average_layer)
  output_layer = Dense(num_out, activation='softmax')(dense_layer)
  for layer in base_model.layers:
      layer.trainable = False
  model = tf.keras.models.Model(inputs=base_model.input, outputs=output_layer)
  return model

model=build_model(2)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

cur_time=datetime.datetime.now()
out_dir=f"runtimes/runtime_{cur_time}"
os.makedirs(out_dir,exist_ok=True)

# Create a ReduceLROnPlateau callback
reduce_lr_callback = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.15,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

model_checkpoint=tf.keras.callbacks.ModelCheckpoint(os.path.join(out_dir,"checkpoint.h5"),save_best_only=True)
history=model.fit(train_dataset,epochs=35,validation_data=test_dataset,callbacks=[model_checkpoint,reduce_lr_callback])

os.makedirs("models",exist_ok=True)
model.save("models/binary_model.h5")

pickle.dump(history,open(os.path.join(out_dir,"model_history.pkl"),"wb"))

pd.DataFrame(history.history).to_csv("model_history.csv")

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
