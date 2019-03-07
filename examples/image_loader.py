import numpy
import PIL
import sys


def load_image(image_name):
  img = PIL.Image.open(image_name)
  img = img.resize((28, 28), resample=PIL.Image.BILINEAR)
  return numpy.asarray(img)