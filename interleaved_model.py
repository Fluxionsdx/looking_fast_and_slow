import tensorflow as tf
from tensorflow.keras import layers


class interleaved_mnLSTM(tf.keras.Model):

    def __init__(self, num_classes=10):
        super(SimpleMLP, self).__init__(name='mlp')
        self.stepNumber = 0
        self.num_classes = num_classes

        self.dense1 = keras.layers.Dense(32, activation='relu')
        self.dense2 = keras.layers.Dense(32, activation='relu')
        self.dense3 = keras.layers.Dense(num_classes, activation='softmax')


    def call(self, inputs):
        if self.stepNumber % 10 == 0:
            x = self.dense1(inputs)
        else:
            x = self.dense2(inputs)
        return self.dense2(x)