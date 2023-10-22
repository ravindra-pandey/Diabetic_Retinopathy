import os,datetime
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print("gpu",gpu,end="/n")
    except RuntimeError as e:
        print(e)

import models as md

image_names = [os.path.join(root, fl) for root, dirs, files in os.walk("augmented_dataset") for fl in files]
np.random.shuffle(image_names)
labels = [0  if i.split("/")[-2] == "normal" else 1  for i in image_names]

X_train,X_test,y_train,y_test=train_test_split(image_names,labels,test_size=0.2,random_state=42)


AUTOTUNE = tf.data.AUTOTUNE
train_dataset=tf.data.Dataset.from_tensor_slices((X_train,y_train)).map(md.load_one_image,num_parallel_calls=AUTOTUNE).cache().batch(batch_size=16).prefetch(AUTOTUNE)
test_dataset=tf.data.Dataset.from_tensor_slices((X_test,y_test)).map(md.load_one_image,num_parallel_calls=AUTOTUNE).cache().batch(batch_size=16).prefetch(AUTOTUNE)

try:
    model=tf.keras.models.load_model(f"runtimes/runtime_2023-10-22 06:29:15.819334/checkpoint.h5")
except:
    model=md.build_model(2)
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

cur_time=datetime.datetime.now()
out_dir=f"runtimes/runtime_{cur_time}"
os.makedirs(out_dir,exist_ok=True)


model_checkpoint=tf.keras.callbacks.ModelCheckpoint(os.path.join(out_dir,"checkpoint.h5"),save_best_only=True)
history=model.fit(train_dataset,epochs=35,validation_data=test_dataset,callbacks=[model_checkpoint,md.reduce_lr_callback])

os.makedirs("models",exist_ok=True)
model.save("models/binary_model.h5")

pickle.dump(history,open(os.path.join(out_dir,"model_history.pkl"),"wb"))

pd.DataFrame(history.history).to_csv("model_history.csv")
md.save_plot(out_dir,history)
