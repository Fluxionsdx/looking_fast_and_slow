import tensorflow as tf 
import os

def load_example(image_path):
       
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize(image, (224,224))
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


def create_datasets():
    train_path = "/storage/imagenette-320/train"
    train_image_list = get_image_list(train_path)
    ds_train = tf.data.Dataset.from_tensor_slices(train_image_list)
    ds_train = ds_train.map(load_example).batch(16).shuffle(1024).repeat()

    val_path = "/storage/imagenette-320/val"
    val_image_list = get_image_list(val_path)
    ds_val = tf.data.Dataset.from_tensor_slices(val_image_list)
    ds_val = ds_val.map(load_example).batch(16)

    
    
    return ds_train, ds_val
