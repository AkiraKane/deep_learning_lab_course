import numpy as np
import math

from ..layers import Layer, Parameterized, Activation

class Conv(Layer, Parameterized):
    """ 
    A convolutional layer that supports convolving an input tensor
    with a set of filters.
    We will, for now, assume that we always want zero padding around
    the input (which we indicate by saying we want a 'same' convolution).
    We assume that the input tensor is shaped:
    (batch_size, input_channels, height, width)
    That is for a 32x32 RGB image and batch size 64 we would have:
    (64, 3, 32, 32)
    
    Parameters
    ----------
    input_layer : a :class:`Layer` instance
    n_feats : the number of features or neurons for the conv layer
    filter_shape : a tuple specifying the filter shape, e.g. (5,5)
    strides : a tuple specifying the stride of the convolution, e.g. (1,1)
    init_stddef : a float specifying the standard deviation for weight init
    padding_mode : a string specifying which padding mode to use 
      (we only support zero padding or 'same' convolutions for now)
    activation_fun : a :class:`Activation` instance
    """
    def __init__(self, input_layer, n_feats,
                 filter_shape, init_stddev, strides=(1,1),
                 padding_mode='same',
                 activation_fun=Activation('relu')):
        """
        Initialize convolutional layer.
        :parameters@param input_layer 
        
        """
        self.n_feats = n_feats
        self.filter_shape = filter_shape
        self.strides = strides
        self.init_stddev = init_stddev
        self.padding_mode = padding_mode
        self.input_shape = input_layer.output_size()
        self.n_channels = self.input_shape[1]
        self.activation_fun = activation_fun
        
        W_shape = (self.n_channels, self.n_feats) + self.filter_shape
        self.W = np.random.normal(size=W_shape, scale=self.init_stddev)
        self.b = np.zeros(self.n_feats)

    def conv(self, input, filters, strides, padding_mode, convout):
        # Pad the input image to realize the 'same convolution'
        filter_shape = filters.shape[-2:]
        pad_row = (filter_shape[0] - 1) / 2.0
        pad_col = (filter_shape[1] - 1) / 2.0
        
        # Pad the images
        input_pad = np.lib.pad(input, ((0, 0), (0, 0), (math.floor(pad_row), math.ceil(pad_row)),
            (math.floor(pad_row), math.ceil(pad_row))), 'constant')

        for img in range(len(input)):
            for output_row in range(input.shape[2]):
                for output_col in range(input.shape[3]):
                    for filter in range(filters.shape[1]):
                        row_start = output_row
                        col_start = output_col
                        row_end = output_row + filter_shape[0]                        
                        col_end = output_col + filter_shape[1]

                        filter_sum = 0
                        for channel in range(filters.shape[0]):
                            filter_sum += np.sum(input_pad[img, channel, row_start : row_end,
                                                       col_start : col_end] * filters[channel,
                                                       filter])

                        convout[img, filter, output_row, output_col] = filter_sum

    def fprop(self, input):
        # we cache the input and the input
        self.last_input = input
        self.input_shape = input.shape
        convout = np.empty(self.output_size())
        self.conv(input, self.W, self.strides, self.padding_mode, convout)
        convout += self.b[np.newaxis, :, np.newaxis, np.newaxis]
        if self.activation_fun is not None:
            return self.activation_fun.fprop(convout)
        else:
            return convout

    def bprop_conv(self,last_input, output_grad_pre, W, input_grad, dW, stride_x, stride_y):
        """
            Parameters:
            - last_input (images, channels, height, width)
            - output_grad_pre (images, n_feats, height, width)
            - W (self.n_channels, self.n_feats, filter_height, filter_width)
            - input_grad (last_input.shape)
            - dW (W.shape)
        """

        # Compute the padding dimensions
        filter_shape = W.shape[-2:]
        pad_row = (filter_shape[0] - 1) / 2.0
        pad_col = (filter_shape[1] - 1) / 2.0
        
        # Pad the images
        input_pad = np.pad(last_input, ((0, 0), (0, 0), (math.floor(pad_row), math.ceil(pad_row)),
                            (math.floor(pad_row), math.ceil(pad_row))), 'constant')
        temp_input_grad = np.zeros(input_pad.shape)

        for image_index in range(input_pad.shape[0]):
            for channel_index in range(input_pad.shape[1]):
                for depth in range(W.shape[1]):
                    for input_row in range(output_grad_pre.shape[2]):
                        for input_column in range(output_grad_pre.shape[3]):
                            # Compute the weight gradient
                            w_pre = output_grad_pre[image_index, depth, input_row, input_column]
                            w_last = input_pad[image_index, channel_index, input_row : input_row + W.shape[2],
                                          input_column : input_column + W.shape[3]]
                            dW[channel_index, depth, :, :] += w_pre * w_last

                            # Compute the input gradient
                            previous_grad = output_grad_pre[image_index, depth, input_row, input_column]
                            previous_weight = W[channel_index, depth, : , :]

                            value = previous_grad * previous_weight

                            input_grad_row_start = input_row
                            input_grad_row_end = input_row + W.shape[2]
                            input_grad_col_start = input_column
                            input_grad_col_end = input_column + W.shape[3]
                            temp_input_grad[image_index, channel_index,
                                            input_grad_row_start : input_grad_row_end,
                                            input_grad_col_start : input_grad_col_end] += value

        input_grad[:,:,:,:] = temp_input_grad[:,:, math.floor(pad_row) : -math.ceil(pad_row),
                                                  math.floor(pad_row) : -math.ceil(pad_row)]
        dW /= last_input.shape[0]

    def bprop(self, output_grad):
        if self.activation_fun is None:
            output_grad_pre = output_grad
        else:
            output_grad_pre = self.activation_fun.bprop(output_grad)
        last_input_shape = self.last_input.shape
        input_grad = np.zeros(last_input_shape)
        self.dW = np.zeros(self.W.shape)
        self.bprop_conv(self.last_input, output_grad_pre, self.W, input_grad,
                        self.dW, self.strides[0], self.strides[1])
        n_imgs = output_grad_pre.shape[0]
        self.db = np.sum(output_grad_pre, axis=(0, 2, 3)) / (n_imgs)
        return input_grad

    def params(self):
        return self.W, self.b

    def grad_params(self):
        return self.dW, self.db
    
    def output_size(self):
        if self.padding_mode == 'same':
            h = self.input_shape[2]
            w = self.input_shape[3]
        else:
            raise NotImplementedError("Unknown padding mode {}".format(self.padding_mode))
        shape = (self.input_shape[0], self.n_feats, h, w)
        return shape


class Pool(Layer):
    """
    A pooling layer for dimensionality reduction.

    Parameters:
    -----------
    input_layer : a :class:`Layer` instance
    n_feats : the number of features or neurons for the conv layer
    pool_shape : a tuple specifying the pooling region size, e.g., (3,3)
    strides : a tuple specifying the stride of the pooling, e.g. (1,1),
       stides == pool_shape results in non-overlaping pooling
    mode : the pooling type (we only support max-pooling for now)
    """
    def __init__(self, input_layer, pool_shape=(3, 3), strides=(1, 1), mode='max'):
        if mode != 'max':
            raise NotImplementedError("Only max-pooling currently implemented")
        self.mode = mode
        self.pool_h, self.pool_w = pool_shape
        self.stride_y, self.stride_x = strides
        self.input_shape = input_layer.output_size()

    def pool(self, input, poolout, switches, pool_h, pool_w, stride_y, stride_x):
        for img in range(len(input)):
            for channel in range(input.shape[1]):
                for out_row in range(0, input.shape[2], stride_y):
                    out_row_end = min(out_row + pool_h, input.shape[2])
                    for out_col in range(0, input.shape[3], stride_x):
                        out_col_end = min(out_col + pool_w, input.shape[3])
                        
                        submatrix = input[img, channel, out_row : out_row_end, out_col : out_col_end]
                        index = np.unravel_index(np.argmax(submatrix), submatrix.shape)
                        absolute_index = index + np.array([out_row, out_col])
                        max_value = submatrix[index]
                        
                        poolout[img, channel, out_row // stride_y, out_col // stride_x] = max_value
                        switches[img, channel, out_row // stride_y, out_col // stride_x] = absolute_index

    def fprop(self, input):
        # we cache the input
        self.last_input_shape = input.shape
        # and also the switches
        # which are the positions were the maximum was
        # we need those for doing the backwards pass!
        self.input_shape = input.shape
        self.last_switches = np.empty(self.output_size()+(2,),
                                      dtype=np.int)
        poolout = np.empty(self.output_size())
        self.pool(input, poolout, self.last_switches, self.pool_h, self.pool_w,
                  self.stride_y, self.stride_x)
        return poolout

    def bprop_pool(self, output_grad, last_switches, input_grad):
        """
            Parameters:
            - output_grad (images, n_feats, height, width)
            - last_switches (output_size + (x, y))
            - input_grad (last_input.shape)
        """

        for image_index in range(output_grad.shape[0]):
            for depth in range(output_grad.shape[1]):
                for input_row in range(output_grad.shape[2]):
                    for input_column in range(output_grad.shape[3]):
                        # Compute the input gradient
                        y = last_switches[image_index, depth, input_row, input_column, 0]
                        x = last_switches[image_index, depth, input_row, input_column, 1]
                        input_grad[image_index, depth, y, x] += output_grad[image_index, depth, input_row, input_column]

    def bprop(self, output_grad):
        input_grad = np.zeros(self.last_input_shape)
        self.bprop_pool(output_grad, self.last_switches, input_grad)
        return input_grad
    
    def output_size(self):
        input_shape = self.input_shape
        shape = (input_shape[0],
                 input_shape[1],
                 input_shape[2]//self.stride_y,
                 input_shape[3]//self.stride_x)
        return shape


class Flatten(Layer):
    """ 
    This is a simple layer that you can use to flatten
    the output from, for example, a convolution or pooling layer. 
    Such that you can put a fully connected layer on top!
    The result will always preserve the dimensionality along
    the zeroth axis (the batch size) and flatten all other dimensions!
    """

    def __init__(self, input_layer):
        self.input_shape = input_layer.output_size()
        
    def fprop(self, input):
        self.last_input_shape = input.shape
        return np.reshape(input, (input.shape[0], -1))

    def bprop(self, output_grad):
        return np.reshape(output_grad, self.last_input_shape)

    def output_size(self):
        return (self.input_shape[0], np.prod(self.input_shape[1:]))
