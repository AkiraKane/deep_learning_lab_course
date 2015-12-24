#!/usr/bin/python2.7
import numpy as np
import nn
import time
# import matplotlib.pyplot as plt

# you can load the cifar10 data like this
data = nn.data.minicifar10()
train_data, valid_data, test_data = data
print("data size: " + str(len(data[0][0])))
train_x, train_y = train_data
valid_x, valid_y = valid_data
test_x, test_y = test_data

train_x = train_x.astype(np.float_) / 255.
train_y = train_y.astype(np.float_) / 255.
valid_x = valid_x.astype(np.float_) / 255.
valid_y = valid_y.astype(np.float_) / 255.
test_x = test_x.astype(np.float_) / 255.
test_y = test_y.astype(np.float_) / 255.

input_shape = (None, 3, 32, 32)
layers = [nn.InputLayer(input_shape)]
layers.append(nn.Conv(
              layers[-1],
              n_feats=2,
              filter_shape=(3, 3),
              init_stddev=0.01,
              activation_fun=nn.layers.Activation('relu')))
layers.append(nn.Flatten(layers[-1]))
layers.append(nn.FullyConnectedLayer(
              layers[-1],
              num_units=100,
              init_stddev=0.1,
              activation_fun=nn.layers.Activation('relu')))
layers.append(nn.FullyConnectedLayer(
              layers[-1],
              num_units=10,
              init_stddev=0.1,
              activation_fun=nn.layers.Activation('relu')))
layers.append(nn.SoftmaxOutput(layers[-1]))
net = nn.NeuralNetwork(layers)

start_train_time = time.time()

net.train(train_x, train_y, Xvalid=valid_x,
          Yvalid=valid_y, learning_rate=0.1,
          max_epochs=20, batch_size=100,
          y_one_hot=True)
end_train_time = time.time()

print("Time: {:.2f}".format(end_train_time - start_train_time))
