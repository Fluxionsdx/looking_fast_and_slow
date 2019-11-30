import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.applications import mobilenet
import pathlib
import hickle as hkl

data_path = "/storage/imagewoof"
data_dir = pathlib.Path(data_path)
CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
IMG_SIZE = 400
def load_example(image_path):
       
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize(image, (IMG_SIZE,IMG_SIZE))
    image = tf.stack([image, image, image])
    image = tf.dtypes.cast(image, dtype="uint8")
    
    parts = tf.strings.split(image_path, "/")  
    label = parts[4] == CLASS_NAMES
    label = tf.stack([label,label,label])
    
    return image, label

def get_image_list(path):
    image_list = []
    for dir in os.listdir(path):
        for img in os.listdir(path + "/" + dir):
            img_path = path + "/" + dir + "/" + img
            image = tf.io.read_file(img_path)
            image = tf.image.decode_jpeg(image)
            if image.shape[2] == 3:
                image_list.append(img_path)
    return image_list


def create_datasets(batch_size, data_path):
    train_path = "/{0}/train".format(data_path)
    train_image_list = get_image_list(train_path)
    ds_train = tf.data.Dataset.from_tensor_slices(train_image_list)
    ds_train = ds_train.map(load_example).batch(batch_size).shuffle(1024).repeat()
    
    val_path = "/{0}/val".format(data_path)
    val_image_list = get_image_list(val_path)
    ds_val = tf.data.Dataset.from_tensor_slices(val_image_list)
    ds_val = ds_val.map(load_example).batch(batch_size)
    
    return ds_train, ds_val, len(train_image_list), len(val_image_list)

def load_datasets_from_disk(batch_size, path):
    train_path = "/{0}/train/np_train.hkl".format(path)
    ds_train = hkl.load(train_path)
    ds_train = tf.data.Dataset.from_tensor_slices(train_image_list)
    ds_train = ds_train.map(load_example).batch(batch_size).shuffle(1024).repeat()
    
    val_path = "/{0}/val/np_val.hkl".format(path)
    ds_val = hkl.load(val_path)
    ds_val = tf.data.Dataset.from_tensor_slices(val_image_list)
    ds_val = ds_val.map(load_example).batch(batch_size)
    

BATCH_SIZE = 32
ds_train, ds_val, train_length, val_length = create_datasets(BATCH_SIZE, data_path)

inputs = tf.keras.layers.Input((None,IMG_SIZE,IMG_SIZE,3))
mn = mobilenet.MobileNet(input_shape=(IMG_SIZE,IMG_SIZE,3), alpha=1.0, include_top=False, weights=None)
net = tf.keras.layers.TimeDistributed(mn, name="mn")(inputs)
#net = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.4))(net)
net = tf.keras.layers.ConvLSTM2D(640,3, return_sequences=True)(net)
net = tf.keras.layers.TimeDistributed(tf.keras.layers.AveragePooling2D(2))(net)
net = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(net)
net = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(10, activation="softmax"))(net)
model = tf.keras.Model(inputs,net)

model.summary()

model.compile(optimizer=tf.keras.optimizers.RMSprop(0.0001),
              loss="categorical_crossentropy",
              metrics=['accuracy']
)

model.fit(ds_train, steps_per_epoch=int(train_length/BATCH_SIZE), epochs=20, validation_data=ds_val)