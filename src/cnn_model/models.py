from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras.models import Sequential
from keras import backend as K


class LeNet():

    def __init__(self, width, height, num_of_classes, depth=3):
        self.model = self.build(depth, num_of_classes)
        self.width = width
        self.height = height

    def build(self, num_of_classes, depth):
        # Initialize the model
        model = Sequential()
        inputShape = (self.height, self.width, depth)

        if (K.image_data_format() == "channels_first"):
            inputShape = (depth, self.height, self.width)
        
        # first set of CONV => RELU => POOL layers
        model.add(
            Conv2D(
                20,
                (5, 5),
                padding='same',
                input_shape=inputShape
        ))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => RELU => POOL layers
        model.add(
            Conv2D(
                50,
                (5, 5),
                padding='same'
            )
        )
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the network architecture
        return model
