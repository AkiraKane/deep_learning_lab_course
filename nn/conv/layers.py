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
                 filter_shape, init_stddev, strides=(1, 1),
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

        W_shape = (self.n_channels, self.n_feats) + self.filter_shape  # noqa
        self.W = np.random.normal(size=W_shape, scale=self.init_stddev)
        self.b = np.zeros(self.n_feats)

    def conv(self, inputs, weights, strides, padding_mode, convout):
        # Pad the input image to realize the 'same convolution'
        filter_dimension = weights.shape[-2:]
        row_pad = (filter_dimension[0] - 1) / 2.0
        col_pad = (filter_dimension[1] - 1) / 2.0

        # Pad the images
        padded = np.pad(inputs, ((0, 0), (0, 0),
                                 (math.floor(row_pad), math.ceil(row_pad)),
                                 (math.floor(col_pad),
                                 math.ceil(col_pad))), 'constant')

        for input in range(len(inputs)):
            for output_row in range(input.shape[2]):
                for output_col in range(input.shape[3]):
                    for filter in range(input.shape[1]):
                        row_start = output_row
                        row_end = output_row + filter_dimension[0]
                        col_start = output_col
                        col_end = output_col + filter_dimension[1]

                        patch_sum = 0
                        for channel in range(weights.shape[0]):
                            patch_sum += np.sum(padded[input, channel,
                                                row_start: row_end,
                                                col_start: col_end] *
                                                weights[channel, filter])

                        convout[input, filter,
                                output_row, output_col] = patch_sum

    def fprop(self, input):
        # we cache the input and the input
        self.last_input = input
        convout = np.empty(self.output_size())
        # TODO
        # This is were you actually do the convolution with W!
        # You do not have to consider the bias!
        # We will simply add it in a second step (see line below)
        # you simply need to convolve the input with self.W and
        # write the result into convout
        # HINT: I recommend putting conv and pooling in little helper functions
        #       at the start of this file!
        #       The call to these should then look something like:
        # conv(input, self.W, self.strides, self.padding_mode, convout)
        # TODO
        convout += self.b[np.newaxis, :, np.newaxis, np.newaxis]
        if self.activation_fun is not None:
            return self.activation_fun.fprop(convout)
        else:
            return convout

    def bprop_conv(self, last_input, output_grad_pre, W, input_grad, dW):  # noqa
        filter_mid_row = W.shape[2] // 2
        filter_mid_col = W.shape[3] // 2

        print("Output-grad shape: " + str(output_grad_pre.shape))
        print("W shape: " + str(W.shape))

        for image_index in range(last_input.shape[0]):
            for channel_index in range(W.shape[1]):
                for row in range(last_input.shape[2]):
                    y_off_min = max(-row, -filter_mid_row)
                    y_off_max = min(last_input.shape[2] -
                                    row, filter_mid_row + 1)

                    for col in range(last_input.shape[3]):
                        convout_grad_value = output_grad_pre[image_index,
                                                             channel_index,
                                                             row, col]
                        x_off_min = max(-col, -filter_mid_col)
                        x_off_max = min(last_input.shape[3] - col,
                                        filter_mid_col + 1)

                        for row_off in range(y_off_min, y_off_max):
                            for col_off in range(x_off_min, x_off_max):
                                img_row = row + row_off
                                img_col = col + col_off
                                fil_row = filter_mid_row + row_off
                                fil_col = filter_mid_col + col_off

                                for img_channel in range(last_input.shape[1]):
                                    input_grad[image_index, img_channel,
                                               img_row,
                                               img_col] += W[img_channel,
                                                             channel_index,
                                                             fil_row, fil_col] \
                                        * convout_grad_value
                                    dW[img_channel, channel_index,
                                       fil_row,
                                       fil_col] += last_input[image_index,
                                                              img_channel,
                                                              img_row, img_col]\
                                        * convout_grad_value

        dW /= last_input.shape[0]

    def bprop(self, output_grad):
        if self.activation_fun is None:
            output_grad_pre = output_grad
        else:
            output_grad_pre = self.activation_fun.bprop(output_grad)
        last_input_shape = self.last_input.shape
        input_grad = np.empty(last_input_shape)
        self.dW = np.empty(self.W.shape)
        # TODO
        # TODO:
        # This is were you have to backpropagate through the convolution
        # and write your results into dW
        # NOTE: again the bias is covered below!
        # HINT: I recommend putting conv and pooling in little helper functions
        #       at the start of this file!
        #       The call to these should then look something like:
        # bprop_conv(self.last_input, output_grad_pre,
        # self.W, input_grad,  self.dW)
        # TODO
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
            raise NotImplementedError(
                "Unknown padding mode {}".format(self.padding_mode))
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

    def __init__(self, input_layer, pool_shape=(3, 3),
                 strides=(1, 1), mode='max'):
        if mode != 'max':
            raise NotImplementedError("Only max-pooling currently implemented")
        self.mode = mode
        self.pool_h, self.pool_w = pool_shape
        self.stride_y, self.stride_x = strides
        self.input_shape = input_layer.output_size()

    def pool(self, input, poolout, last_switches,
             pool_h, pool_w, stride_y, stride_x):
        for img in range(len(input)):
            for channel in range(input.shape[1]):
                for out_row in range(0, input.shape[2], stride_y):
                    out_row_end = min(out_row + pool_h, input.shape[2])
                    for out_col in range(0, input.shape[3], stride_x):
                        out_col_end = min(out_col + pool_w, input.shape[3])

                        submatrix = input[img, channel, out_row:
                                          out_row_end, out_col: out_col_end]

                        max_value = submatrix.max()
                        max_coords = np.ravel(np.where(submatrix == max_value))
                        max_coords += np.array([out_row, out_col])

                        poolout[img, channel, out_row // stride_y,
                                out_col // stride_x] = max_value
                        last_switches[img, channel, out_row // stride_y,
                                      out_col // stride_x] = max_coords

    def fprop(self, input):
        # we cache the input
        self.last_input_shape = input.shape
        # and also the switches
        # which are the positions were the maximum was
        # we need those for doing the backwards pass!
        self.last_switches = np.empty(self.output_size() + (2,),
                                      dtype=np.int)
        poolout = np.empty(self.output_size())
        # TODO
        # this is were you have to implement pooling
        # HINT: it is very similar to the convolution from above
        #       only that you compute a max rather than a multiplication with
        #       weights
        # HINT: You should store the result in poolout and the max positions
        #       (switches) in self.last_switches, you will need those in the
        #       backward pass!
        # the call should look something like:
        self.pool(input, poolout, self.last_switches, self.pool_h, self.pool_w,
                  self.stride_y, self.stride_x)
        # TODO
        return poolout

    def bprop(self, output_grad):
        input_grad = np.empty(self.last_input_shape)
        # TODO
        # implement the backward pass through the pooling
        # it should use the switches, the call should look something like:
        # bprop_pool(output_grad, self.last_switches, input_grad)
        # TODO
        return input_grad

    def output_size(self):
        input_shape = self.input_shape
        shape = (input_shape[0],
                 input_shape[1],
                 input_shape[2] // self.stride_y,
                 input_shape[3] // self.stride_x)
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
