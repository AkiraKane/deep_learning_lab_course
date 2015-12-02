#!/usr/bin/python3
"""
Execise 2a of deep learning lab course
"""

import numpy as np
import math

# import nn


def conv(image, filters, convout):
    """
    TODO: implement convolution here
    """
    pad_dim_x = (filters.shape[2] - 1) / 2
    pad_dim_y = (filters.shape[3] - 1) / 2
    padded_image = np.pad(image, ((0, 0), (0, 0),
                          (math.floor(pad_dim_x), math.ceil(pad_dim_x)),
                          (math.floor(pad_dim_y), math.ceil(pad_dim_y))),
                          'constant')
    for batch in range(image.shape[0]):
        for height in range(image.shape[2]):
            for width in range(image.shape[3]):
                image_part = padded_image[batch, 0,
                                          height:(height + filters.shape[2]),
                                          width:(width + filters.shape[3])]
                image_solution = np.multiply(filters[0][0], image_part)
                convout[batch, 0, height, width] = np.sum(image_solution)


def main():
    """
    Entrypoint if called as an executable
    """
    img = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12],
                      [13, 14, 15, 16]], dtype=np.float64).reshape(1, 1, 4, 4)
    # remember the first dimension is the batch size here
    # lets repeat the image so that we get a more useful test
    imgs = np.repeat(img, 2, axis=0)
    print(imgs.shape)

    # to test we will only use one (3,3) filter which does not fits into the
    # (4,4) images so you need to do padding ;) first dimension is input
    # channels, second is the number of filters 3rd
    # and 4th are filter dimensions
    filters = np.eye(3).reshape((1, 1, 3, 3))
    print(filters.shape)

    # since we are doing same convolutions the output should be the same size
    # as the input
    convout = np.zeros_like(imgs)

    # apply the convolution
    conv(imgs, filters, convout)

    # print the output and compare to the desired output
    print(convout)

    real_output = np.asarray(
        [[[[7., 9., 11., 4.],
           [15., 18., 21., 11.],
           [23., 30., 33., 19.],
           [13., 23., 25., 27.]]],


         [[[7., 9., 11., 4.],
           [15., 18., 21., 11.],
           [23., 30., 33., 19.],
           [13., 23., 25., 27.], ]]], dtype=np.float64)

    diff = np.linalg.norm(real_output - convout)
    # the difference between those should be smaller than eps
    eps = 1e-4
    print("Diff {}".format(diff))
    assert diff < eps

if __name__ == '__main__':
    main()
