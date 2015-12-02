#!/usr/bin/python3

import numpy as np
import nn


def main():
    input_shape = (5, 1, 28, 28)
    n_labels = 6
    layers = [nn.InputLayer(input_shape)]

    layers.append(nn.Conv(
        layers[-1],
        n_feats=2,
        filter_shape=(3, 3),
        init_stddev=0.01,
        activation_fun=nn.Activation('relu'),
    ))
    layers.append(nn.Pool(layers[-1]))
    layers.append(nn.Flatten(layers[-1]))
    layers.append(nn.FullyConnectedLayer(
        layers[-1],
        num_units=6,
        init_stddev=0.1,
        activation_fun=None
    ))
    layers.append(nn.SoftmaxOutput(layers[-1]))
    net = nn.NeuralNetwork(layers)

    # create random data
    X = np.random.normal(size=input_shape)  # noqa
    Y = np.zeros((input_shape[0], n_labels))  # noqa
    for i in range(Y.shape[0]):
        idx = np.random.randint(n_labels)
        Y[i, idx] = 1.
    # perform gradient checking, this should go through if you implemented
    # everything correctly!
    net.check_gradients(X, Y)

if __name__ == '__main__':
    main()
