import tensorflow as tf
import numpy as np
import os
import struct


from CAR import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'#Suppress some useless warnings related to CPU support of Tensorflow Installation. Do I run on GPU?

def multiple_1d_convolution():
    tf.reset_default_graph()

    tf.reset_default_graph()

    model = tf.keras.models.Sequential()

    ############################################### Layer 0 ##########################################

    kernel_size_1d_0 = 50#Adjust? (Doesn't affect next layer size)
    n_kernels_0 = 32#More features in next layer (32 as in AlexNet)
    stride_1d_0 = 4#Decreasing size of whole "perceptive field" of all neurons (4 as in AlexNet)

    model.add(tf.keras.layers.Conv2D(n_kernels_0, (kernel_size_1d_0, 1), input_shape=(20000,10, 1),
                                     padding = "SAME", strides=[stride_1d_0,1],
                                     activation = tf.keras.activations.relu))
    #We have 128 versions of kernels for each of the 20.000 positions for each sensor seperately. Hence the shape:
    print(model.layers[-1].output_shape)

    ############################################### Layer 1 ##########################################
    kernel_size_1d_1 = 50#Adjust? (Doesn't affect next layer size)
    n_kernels_1 = 96#More features in next layer
    stride_1d_1 = 2#Decreasing size of whole "perceptive field" of all neurons (2 as in AlexNet)

    model.add(tf.keras.layers.Conv2D(n_kernels_1, (kernel_size_1d_1, 1),
                                     padding = "SAME", strides=[stride_1d_1,1],
                                     activation = tf.keras.activations.relu))

    print(model.layers[-1].output_shape)

    """############################################### Layer 2 ##########################################
    kernel_size_1d_2 = 100#Adjust? (Doesn't affect next layer size)
    n_kernels_2 = 128#More features in next layer
    stride_1d_2 = 2#Decreasing size of whole "perceptive field" of all neurons

    model.add(tf.keras.layers.Conv2D(n_kernels_2, (kernel_size_1d_2, 1),
                                     padding = "SAME", strides=[stride_1d_2,1],
                                     activation = tf.keras.activations.relu))

    print(model.layers[-1].output_shape)


    ############################################### Layer 3 ##########################################
    kernel_size_1d_3 = 100#Adjust? (Doesn't affect next layer size)
    n_kernels_3 = 256#More features in next layer
    stride_1d_3 = 3#Decreasing size of whole "perceptive field" of all neurons

    model.add(tf.keras.layers.Conv2D(n_kernels_3, (kernel_size_1d_3, 1),
                                     padding = "SAME", strides=[stride_1d_3,1],
                                     activation = tf.keras.activations.relu))

    print(model.layers[-1].output_shape)

    ############################################### Layer 4 ##########################################
    kernel_size_1d_4 = 100#Adjust? (Doesn't affect next layer size)
    n_kernels_4 = 256#More features in next layer
    stride_1d_4 = 1#Decreasing size of whole "perceptive field" of all neurons

    model.add(tf.keras.layers.Conv2D(n_kernels_4, (kernel_size_1d_3, 1),
                                     padding = "SAME", strides=[stride_1d_3,1],
                                     activation = tf.keras.activations.relu))

    print(model.layers[-1].output_shape)

    ############################################### Conv Layer 0##########################################

    model.add(tf.keras.layers.Flatten())
    model.add(tf.layers.Dense(4096, activation=tf.keras.activations.relu))
    print(model.layers[-1].output_shape)

    ############################################### Conv Layer 1##########################################"""

    model.add(tf.keras.layers.Flatten())
    model.add(tf.layers.Dense(2048, activation=tf.keras.activations.relu))
    print(model.layers[-1].output_shape)

    ############################################### Conv Layer 2##########################################

    model.add(tf.keras.layers.Flatten())
    model.add(tf.layers.Dense(3, activation=tf.keras.activations.relu))
    print(model.layers[-1].output_shape)


    model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adadelta(lr=0.1),
              metrics=['accuracy'])

    return model


def train(epochs = 1, batch_size=10):
    car = CAR("/net/projects/scratch/summer/valid_until_31_January_2020/ann4auto/Combined_Chunks/Acc_Vel/1.0s_duration/0.0s_overlap")

    step = 1
    for epoch in range(epochs):

        print('Epoch = ', epoch + 1)
        prelim_acc = 0

        for batch, label_batch in mnist.get_training_batch(batch_size = 10):

	    #we use expand_dims to make our 2d arrays of data a 3d array [[[1]],[[2]]] instead of [[1,2],[2,2]] such that it looks like an image with one colorchannel

            stat = model.fit(np.expand_dims(batch,axis=3),label_batch, verbose = 0)#verbose =0 supresses output

            prelim_acc += stat.history["acc"][0]

            #Output progress here:
            if (step % 100)==0:
                print(prelim_acc/100)
                prelim_acc = 0

            step+=1

    test_source, test_target = next(mnist.get_validation_batch(batch_size = 1000))
    model.evaluate(np.expand_dims(test_source,axis=3),test_target)


if __name__ == "__main__":
    model = multiple_1d_convolution()
    train()
