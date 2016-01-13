"""
    Copyright 2016 University of Freiburg
    Max Lohmann <Max.Lohmann@uranus.uni-freiburg.de>
    Janosch Deurer <deurerj@tf.uni-freiburg.de>
"""
import numpy as np
import nn
import time
from augmentation import Transformer

# Load CIFAR-10 data
data = nn.data.minicifar10()
train_data, valid_data, test_data = data
print("Data size before augmentation: " + str(len(data[0][0])))
train_x, train_y = train_data
valid_x, valid_y = valid_data
test_x, test_y = test_data

# Do the augmentation
t = Transformer()
ori_size = len(train_x)
for i in range(ori_size):
    image = train_x[i]
    label = train_y[i]

    transformed = t.process(image, n=2)
    train_x = np.append(train_x, transformed, axis=0)
    train_y = np.append(train_y, [label] * len(transformed), axis=0)

print("Data size after augmentation: " + str(train_x.shape))

# Normalize the CIFAR-10 data
train_x = train_x.astype(np.float_) / 255.
train_y = train_y.astype(np.float_)
valid_x = valid_x.astype(np.float_) / 255.
valid_y = valid_y.astype(np.float_)
test_x = test_x.astype(np.float_) / 255.
test_y = test_y.astype(np.float_)

# Define the network structure
input_shape = (None, 3, 32, 32)
layers = [nn.InputLayer(input_shape)]
layers.append(nn.Conv(
              layers[-1],
              n_feats=32,
              filter_shape=(5, 5),
              init_stddev=0.1,
              activation_fun=nn.layers.Activation('relu')))

layers.append(nn.Pool(layers[-1]))
              
layers.append(nn.Conv(
              layers[-1],
              n_feats=32,
              filter_shape=(5, 5),
              init_stddev=0.1,
              activation_fun=nn.layers.Activation('relu')))

layers.append(nn.Pool(layers[-1]))

layers.append(nn.Conv(
              layers[-1],
              n_feats=64,
              filter_shape=(5, 5),
              init_stddev=0.1,
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
              activation_fun=None))
layers.append(nn.SoftmaxOutput(layers[-1]))

net = nn.NeuralNetwork(layers)

# Start the actual training
start_train_time = time.time()
net.train(train_x, train_y, Xvalid=valid_x,
          Yvalid=valid_y, learning_rate=0.01,
          max_epochs=100, batch_size=128,
          y_one_hot=True)
end_train_time = time.time()

# Print the results
print("Training took: {:.2f} seconds".format(end_train_time - start_train_time))