#!/usr/bin/python3

import numpy as np


def pool(imgs, poolout, switches, pool_h, pool_w, stride_y, stride_x):
    """
    Parameters:
    -----------
    imgs: input tensor of size (batch_size, chan_in, height, width)
    poolout: the output tensor of size (batch_size, chan_in, height//stride_y,
    width//stride_x) switches: binary encoding of maximum positions, we store
    them in a tensor of size (batch_size, chan_in, height//stride_y,
    width//stride_x, 2), where the last two dimensions are used to specify y
    and x positions of the maximum element!
    pool_h: the height of the pooling regions
    pool_w: the width of the pooling regions
    stride_y: the step size in y direction (e.g. if you want non-overlapping
    pooling set stride_y = pool_h) stride_x: the step size in x direction

    """
    # TODO: implement pooling here
    for img in range(len(imgs)):
        for channel in range(imgs.shape[1]):
            for out_row in range(0, imgs.shape[2], stride_y):
                out_row_end = min(out_row + pool_h, imgs.shape[2])
                for out_col in range(0, imgs.shape[3], stride_x):
                    out_col_end = min(out_col + pool_w, imgs.shape[3])

                    submatrix = imgs[img, channel,
                                     out_row: out_row_end,
                                     out_col: out_col_end]

                    max_value = submatrix.max()
                    max_coords = np.ravel(np.where(submatrix == max_value))
                    max_coords += np.array([out_row, out_col])

                    poolout[img, channel, out_row // stride_y,
                            out_col // stride_x] = max_value
                    switches[img, channel, out_row // stride_y,
                             out_col // stride_x] = max_coords


def main():
    img = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [
                     13, 14, 15, 16]], dtype=np.float64).reshape(1, 1, 4, 4)
    # remember the first dimension is the batch size here
    # lets repeat the image so that we get a more useful test
    imgs = np.repeat(img, 2, axis=0)
    print(imgs.shape)
    # to test we will pool in 2x2 regions with stride 2
    img_h = img.shape[2]
    img_w = img.shape[3]
    stride_y, stride_x = 2, 2
    pool_h, pool_w = 2, 2
    # this gives us output size
    poolout_h = img_h // stride_y
    poolout_w = img_w // stride_x
    # since we are doing same convolutions the output should be the same size
    # as the input
    poolout = np.zeros((imgs.shape[0], imgs.shape[1], poolout_h, poolout_w))
    # also create storage for the switches
    switches = np.zeros(poolout.shape + (2,), dtype=np.int)
    print(poolout.shape)
    print(switches.shape)

    # apply the pooling
    pool(imgs, poolout, switches, pool_h, pool_w, stride_y, stride_x)

    # print the output and compare to the desired output
    print(poolout)

    real_output = np.asarray(
        [[[[6.,   8.],
           [14.,  16.]]],
            [[[6.,   8.],
              [14.,  16.]]]], dtype=np.float64)

    diff = np.linalg.norm(real_output - poolout)
    # the difference between those should be smaller than eps
    eps = 1e-4
    print("Diff {}".format(diff))
    assert(diff < eps)

    # we can also take a look at the switches
    print(switches)

if __name__ == '__main__':
    main()
