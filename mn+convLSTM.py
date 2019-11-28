import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from classification_dataset_builder import *

#tf.enable_eager_execution()

inputs = layers.Input((None,224,224,3))
mobilenet = tf.keras.applications.mobilenet.MobileNet(input_shape=(224,224,3), include_top=False, weights='imagenet')
net = layers.TimeDistributed(mobilenet, name="mn")(inputs)
net = layers.ConvLSTM2D(320, 3, return_sequences=True)(net)
net = layers.TimeDistributed(layers.AveragePooling2D())(net)
net = layers.TimeDistributed(layers.Flatten())(net)
net = layers.TimeDistributed(layers.Dense(1, activation="softmax"))(net)
model = Model(inputs,net)

#for layer in mobilenet.layers:
#    if "_bn" in layer.name:
#    	print(layer.name)
#    	layer.trainable = False


tensorboard = tf.keras.callbacks.TensorBoard(log_dir='.', histogram_freq=0,
                          write_graph=True, write_images=False)

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss="binary_crossentropy",
              metrics=['accuracy']
)
model.summary()


ds_train, ds_test = create_datasets('cats_vs_dogs')


#need to add code to calculate steps_per_epoch and validation steps

model.fit(ds_train, epochs=3, steps_per_epoch=1309 ,validation_data=ds_test, validation_steps=131, callbacks=[tensorboard])

model.save("./saved_model")