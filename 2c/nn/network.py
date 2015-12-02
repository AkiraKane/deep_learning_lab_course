import numpy as np
from .layers import Layer, Parameterized, Activation, one_hot, unhot
import math
import time

class NeuralNetwork:
    """ Our Neural Network container class.
    """
    def __init__(self, layers):
        self.layers = layers
        
    def _loss(self, X, Y):
        Y_pred = self.predict(X)
        return self.layers[-1].loss(Y, Y_pred)

    def predict(self, X):
        """ Calculate an output Y for the given input X. """
        h_in = X
        for layer in self.layers:
            h_in = layer.fprop(h_in)
            #print("Prediction-shape of layer %s: %s" % (str(layer), str(h_in.shape)))
        Y_pred = h_in
        return Y_pred
    
    def backpropagate(self, Y, Y_pred, upto=0):
        """ Backpropagation of partial derivatives through 
            the complete network up to layer 'upto'
        """
        next_grad = self.layers[-1].input_grad(Y, Y_pred)
        for layer in reversed(self.layers[upto:-1]):
            next_grad = layer.bprop(next_grad)
        return next_grad
    
    def classification_error(self, X, Y):
        """ Calculate error on the given data 
            assuming they are classes that should be predicted. 
        """
        Y_pred = unhot(self.predict(X))
        error = Y_pred != Y
        return np.mean(error)
    
    def sgd_epoch(self, X, Y, learning_rate, batch_size):
        n_samples = X.shape[0]
        n_batches = n_samples // batch_size
        for b in range(n_batches):
            batch_begin = b*batch_size
            batch_end = batch_begin+batch_size
            X_batch = X[batch_begin:batch_end]
            Y_batch = Y[batch_begin:batch_end]

            #print("Batch %d: %s" % (b, str(X_batch.shape)))

            # Forward propagation
            Y_pred = self.predict(X_batch)

            # Back propagation
            self.backpropagate(Y_batch, Y_pred)

            # Update parameters
            for layer in self.layers:
                if isinstance(layer, Parameterized):
                    for param, grad in zip(layer.params(),
                                          layer.grad_params()):
                        param -= learning_rate*grad
    
    def gd_epoch(self, X, Y, learning_rate):
        self.sgd_epoch(X, Y, learning_rate, X.shape[0])
    
    def train(self, X, Y, Xvalid=None, Yvalid=None, 
              learning_rate=0.1, max_epochs=100, 
              batch_size=64, descent_type="sgd", y_one_hot=True):
        """ Train network on the given data. """
        print("Train")
        n_samples = X.shape[0]
        n_batches = n_samples // batch_size
        if y_one_hot:
            Y_train = one_hot(Y)
        else:
            Y_train = Y
        print("... starting training")

        train_errors = list()
        valid_errors = list()

        for e in range(max_epochs+1):
            start_time = time.time()
            if descent_type == "sgd":
                self.sgd_epoch(X, Y_train, learning_rate, batch_size)
            elif descent_type == "gd":
                self.gd_epoch(X, Y_train, learning_rate)
            else:
                raise NotImplemented("Unknown gradient descent type {}".format(descent_type))

            # Output error on the training data
            #print("Pre loss")
            train_loss = self._loss(X, Y_train)
            #print("Post loss")
            train_error = self.classification_error(X, Y)
            train_errors.append(train_error)
            print('epoch {:4d}, loss {:.4f}, train error {:.4f}'.format(e, train_loss, train_error))
            if Xvalid is not None:
                valid_error = self.classification_error(Xvalid, Yvalid)
                valid_errors.append(valid_error)
                print('\t\t\t valid error {:.4f}'.format(valid_error))

            end_time = time.time()
            print("Time taken: {:.2f}s".format(end_time - start_time))

        return (train_errors, valid_errors)
    
    def check_gradients(self, X, Y):
        """ Helper function to test the parameter gradients for
        correctness. """
        for l, layer in enumerate(self.layers):
            if isinstance(layer, Parameterized):
                print('checking gradient for layer {}'.format(l))
                for p, param in enumerate(layer.params()):
                    # we iterate through all parameters
                    param_shape = param.shape
                    # define functions for conveniently swapping
                    # out parameters of this specific layer and 
                    # computing loss and gradient with these 
                    # changed parametrs
                    def output_given_params(param_new):
                        """ A function that will compute the output 
                            of the network given a set of parameters
                        """
                        # copy provided parameters
                        param[:] = np.reshape(param_new, param_shape)
                        # return computed loss
                        return self._loss(X, Y)

                    def grad_given_params(param_new):
                        """A function that will compute the gradient 
                           of the network given a set of parameters
                        """
                        # copy provided parameters
                        param[:] = np.reshape(param_new, param_shape)
                        # Forward propagation through the net
                        Y_pred = self.predict(X)
                        # Backpropagation of partial derivatives
                        self.backpropagate(Y, Y_pred, upto=l)
                        # return the computed gradient 
                        return np.ravel(self.layers[l].grad_params()[p])

                    # let the initial parameters be the ones that
                    # are currently placed in the network and flatten them
                    # to a vector for convenient comparisons, printing etc.
                    param_init = np.ravel(np.copy(param))
                    
                    # ####################################
                    #      compute the gradient with respect to
                    #      the initial parameters in two ways:
                    #      1) with grad_given_params()
                    #      2) with finite differences 
                    #         using output_given_params()
                    #         (as discussed in the lecture)
                    #      if your implementation is correct 
                    #      both results should be epsilon close
                    #      to each other!
                    # ####################################
                    epsilon = 1e-8
                    loss_base = output_given_params(param_init)
                    gparam_bprop = grad_given_params(param_init)
                    gparam_fd = np.zeros_like(param_init)
                    for i in range(len(param_init)):
                        param_init[i] += epsilon
                        gparam_fd[i] = (output_given_params(param_init) - loss_base) / (epsilon)
                        if math.isnan(gparam_fd[i]) or math.isinf(gparam_fd[i]):
                            print("---------")
                            print("Output given params: \n" + str(output_given_params(param_init)))
                            print("Loss base: \n" + str(loss_base))
                            print("Epsilon: \n" + str(epsilon))

                        param_init[i] -= epsilon
                    #print("gparam_fd: \n" + str(gparam_fd))
                    err = np.mean(np.abs(gparam_bprop - gparam_fd))
                    #print(gparam_bprop / gparam_fd)
                    print('diff {:.2e}'.format(err))
                    #assert(err < 20. * epsilon)
                    
                    # reset the parameters to their initial values
                    param[:] = np.reshape(param_init, param_shape)
