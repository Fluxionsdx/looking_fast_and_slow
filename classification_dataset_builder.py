import tensorflow as tf 
import tensorflow_datasets as tfds

def load_example(example):
    # need to modify this function to have variable number of timesteps
    example["image"] = tf.image.resize(example["image"], (224,224))
    example["image"] = tf.stack([example["image"], example["image"], example["image"]])
    
    example["label"] = tf.stack([example["label"],example["label"],example["label"]])
    example["label"] = tf.expand_dims(example["label"], axis=-1)
    
    data = example["image"]
    label = example["label"]

    return data, label


def create_datasets(name):
    ds_train = tfds.load(name=name, split=tfds.Split.TRAIN.subsplit(tfds.percent[:90]))
    ds_train = ds_train.map(load_example).repeat().batch(16)
    #ds_train = ds_train.batch(16)
    #ds_train = ds_train.repeat()

    ds_test = tfds.load(name=name, split=tfds.Split.TRAIN.subsplit(tfds.percent[91:]))
    ds_test = ds_test.map(load_example)
    ds_test = ds_test.batch(16)
    ds_test = ds_test.repeat()
    
    
    return ds_train, ds_test

