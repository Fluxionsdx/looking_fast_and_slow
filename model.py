

def mnLSTM():  
    

    inputs = tf.keras.layers.Input((None,224,224,3))
    mobilenet = mobilenet.MobileNet(input_shape=(224,224,3), alpha=0.5, include_top=False, weights='imagenet')
    net = layers.TimeDistributed(mobilenet, name="mn")(inputs)
    net = layers.ConvLSTM2D(50,3, return_sequences=True)(net)
    net = layers.TimeDistributed(layers.Flatten())(net)
    net = layers.TimeDistributed(layers.Dense(10, activation="softmax"))(net)
    model = Model(inputs,net)
    
    return model

    