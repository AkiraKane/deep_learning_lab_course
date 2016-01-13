"""
    Copyright 2016 University of Freiburg
    Max Lohmann <Max.Lohmann@uranus.uni-freiburg.de>
    Janosch Deurer <deurerj@tf.uni-freiburg.de>
"""
import scipy.ndimage as sc
import numpy as np
from PIL import Image
import random
import sys
import math

class Transformer:
  def __init__(self):
    random.seed()

  def color_value(self, arr, x, y, default_color=np.zeros(3)):
    shape = arr.shape
    if x < 0: return default_color
    if x >= shape[0]: return default_color
    if y < 0: return default_color
    if y >= shape[1]: return default_color
    return arr[x, y]

  def rotate(self, arr, angle):
    return sc.interpolation.rotate(arr, angle, reshape=False)

  def shift(self, arr, offset):
    tmp_arr = np.zeros(arr.shape)
    for row in range(arr.shape[0]):
      for col in range(arr.shape[1]):
        tmp_arr[row, col] = self.color_value(arr, row + offset[0], col + offset[1])
    return tmp_arr

  def zoom(self, arr, factor):
    zoomed = sc.interpolation.zoom(arr, (factor, factor, 1), prefilter=False, order=1)
    return self.crop(zoomed, arr.shape[0], arr.shape[1])

  def crop(self, arr, height, width):
    if arr.shape[0] < height or arr.shape[1] < width:
      pad_up = math.ceil((height - arr.shape[0]) / 2.)
      pad_down = math.floor((height - arr.shape[0]) / 2.)
      pad_left = math.ceil((width - arr.shape[1]) / 2.)
      pad_right = math.floor((width - arr.shape[1]) / 2.)
      return np.pad(arr, ((pad_up, pad_down), (pad_left, pad_right),(0,0)), mode="constant")
    tmp_arr = np.zeros((height, width, arr.shape[2]), dtype=int)
    center_x = math.floor(arr.shape[0]/2.)
    center_y = math.floor(arr.shape[1]/2.)
    tmp_arr[0:height, 0:width] = arr[center_x - math.floor(height/2.):center_x + math.ceil(height/2.), center_y - math.floor(width/2.):center_y + math.ceil(width/2.)]
    return tmp_arr

  def mirror(self, arr):
    if (random.random() > 0.5): return np.flipud(arr)
    return np.fliplr(arr)

  def process(self, arr, n=2):
    input = np.reshape(arr, (32,32,3))
    output = []
    for i in range(n):
      temp = np.copy(input)
      if (random.random() > 0.5):
        temp = self.zoom(temp, random.uniform(0.66, 1.5))
      if (random.random() > 0.5):
        temp = self.rotate(temp, random.uniform(-90, 90))
      if (random.random() > 0.5):
        temp = self.shift(temp, (random.uniform(-8, 8), random.uniform(-8, 8)))      
      if (random.random() > 0.5):
        temp = self.mirror(temp)
      output.append(np.reshape(temp, (3,32,32)))
    return output

  def process_zoom(self, arr, n=2):
    input = np.reshape(arr, (32,32,3))
    output = []
    for i in range(n):
      temp = np.copy(input)
      temp = self.zoom(temp, random.uniform(0.66, 1.5))
      output.append(np.reshape(temp, (3,32,32)))
    return output

  def process_rotate(self, arr, n=2):
    input = np.reshape(arr, (32,32,3))
    output = []
    for i in range(n):
      temp = np.copy(input)
      temp = self.rotate(temp, random.uniform(-90, 90))
      output.append(np.reshape(temp, (3,32,32)))
    return output

  def process_shift(self, arr, n=2):
    input = np.reshape(arr, (32,32,3))
    output = []
    for i in range(n):
      temp = np.copy(input)
      temp = self.shift(temp, (random.uniform(-8, 8), random.uniform(-8, 8)))     
      output.append(np.reshape(temp, (3,32,32)))
    return output

  def process_mirror(self, arr, n=2):
    input = np.reshape(arr, (32,32,3))
    output = []
    for i in range(n):
      temp = np.copy(input)
      temp = self.mirror(temp)
      output.append(np.reshape(temp, (3,32,32)))
    return output

  def process_all(self, arr):
    input = np.reshape(arr, (32,32,3))
    output = []
    temp = np.copy(input)
    temp = self.mirror(temp)
    output.append(np.reshape(temp, (3,32,32)))
    temp = np.copy(input)
    temp = self.zoom(temp, random.uniform(0.66, 1.5))
    output.append(np.reshape(temp, (3,32,32)))
    temp = np.copy(input)
    temp = self.rotate(temp, random.uniform(-90, 90))
    output.append(np.reshape(temp, (3,32,32)))
    temp = np.copy(input)
    temp = self.shift(temp, (random.uniform(-8, 8), random.uniform(-8, 8)))
    output.append(np.reshape(temp, (3,32,32)))
    return output