"""
This module performs pre-processing on the images
that serve as input to the neural network.
"""
from glob import glob
import os
from os.path import isfile, join
import tensorflow as tf
from PIL import Image
import csv
from constants import NUM_THREADS, DIM_X, DIM_Y, DIM_Z, Y_DIM


def get_bbbc021_filenames(base_path, csv_path):
    file = open(csv_path, "r")
    reader = csv.reader(file)
    next(reader, None)  # skip the headers
    paths = []
    for line in reader:
        image_DAPI_path = base_path + line[2]
        image_tubulin_path = base_path + line[4]
        image_actin_path = base_path + line[6]
        paths.append([image_DAPI_path, image_tubulin_path, image_actin_path])
    return paths
        
def read_bbbc021_images(filepath_packed):
    image_DAPI_path = filepath_packed[0]
    image_tubulin_path = filepath_packed[1]
    image_actin_path = filepath_packed[2]
    image_DAPI, _ = read_image(image_DAPI_path)
    image_tubulin, _ = read_image(image_tubulin_path)
    image_actin, _ = read_image(image_actin_path)
    rgb_image = tf.concat([image_DAPI, image_tubulin, image_actin,], axis=2)
    return transform_image(rgb_image)

def read_image(filename):
    """
    This function reads and resizes the image.
    """
    image_string = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=1)
    return tf.image.convert_image_dtype(image, tf.float32), filename
    
def transform_image(image):    
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    #image = random_rotation(image)
    #celebA
    #if (DIM_X != 256):
      #image = tf.expand_dims(image, 0)
      #image = tf.image.resize_bilinear(image, [128,128])
      #image = tf.squeeze(image, 0)
    image = tf.image.resize_with_crop_or_pad(image, 206, 256)
    image = tf.random_crop(image, [128,128,3])
    image = tf.image.resize(image, [256,256])
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image

def random_rotation(image):
    """
    This function adds random rotation for dataset augmentation.
    """
    k = tf.random_uniform(shape=[], minval=0, maxval=3, dtype=tf.int32)
    return tf.image.rot90(image, k)
  
    
def labels_from_filenames(filenames):
  """
  This function returns the data labels in one-hot format.
  """
  labels = [1 if "pos" in file.split("/")[-1] else 0 for file in filenames]
  labels_one_hot = tf.one_hot(labels, Y_DIM, 1.0, 0.0)  
  return labels_one_hot
  
    
def create_dataset(path, batch_size, num_epochs):
    """
    This function creates a dataset from a path
    batch_size: number of samples before updating weights
    num_epochs: number of times to go through the entire dataset
    """
    #convert_tiff_to_jpeg(path)
    
    #celebA
    #filenames = glob(os.path.join(path, "*.jpeg"))
    #filenames.extend(glob(os.path.join(path, "*.jpg")))
    
    #histology
    #filenames = glob(os.path.join(path, "*.jpg"))
    
    #bbbc021
    filenames = get_bbbc021_filenames(path, path + "BBBC021_v1_image.csv")
    
    
    dataset = (tf.data.Dataset.from_tensor_slices(filenames)
                .apply(tf.data.experimental.shuffle_and_repeat(buffer_size=len(filenames), count=num_epochs))
                .map(read_bbbc021_images)
                .apply(tf.contrib.data.ignore_errors())
                .batch(batch_size)
                .prefetch(1))

    return dataset, len(filenames)


def extract_patches(image, patch_size):
    """ 
    This function extracts square patches of 'patch_size' from the image
    """

    height = patch_size
    width = patch_size
    strides_rows = 500
    strides_cols = 500

    # The size of sliding window
    ksizes = [1, height, width, 1]
    strides = [1, strides_rows, strides_cols, 1]
    rates = [1, 1, 1, 1] # sample pixel consecutively

    image = tf.expand_dims(image, 0)
    image_patches = tf.extract_image_patches(image, ksizes, strides, rates, 'VALID')
    return tf.reshape(image_patches, [-1, height, width, 3])


def convert_tiff_to_jpeg(path):
    """
    This function converts .tif or .png to .jpg image format.
    """
    filenames = [f for f in os.listdir(path) if isfile(join(path, f))]
    for filename in filenames:
        if os.path.splitext(filename)[1].lower() == ".tif" or os.path.splitext(filename)[1].lower() == ".png":
            if os.path.isfile(os.path.splitext(os.path.join(path, filename))[0] + ".jpg"):
                print("A jpeg file already exists for %s" % filename)
            else:
                outputfile = os.path.splitext(filename)[0] + ".jpg"
                try:
                    image = Image.open(os.path.join(path, filename))
                    print("Converting jpeg for %s" % filename)
                    image.thumbnail(image.size)
                    image.save(os.path.join(path, outputfile), "JPEG", quality=100)
                except IOError as err:
                    print("I/O error({0}): {1}".format(err.errno, err.strerror))
