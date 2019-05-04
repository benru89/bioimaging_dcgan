"""This module does blah blah."""
from glob import glob
import os
from os.path import isfile, join
import tensorflow as tf
from PIL import Image
from constants import NUM_THREADS



def read_image(filename, channels):
    """
    This function does blah blah.
    """
    image_string = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=channels)
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    # image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_flip_up_down(image)
    # image = random_rotation(image)
    # image = tf.image.resize_images(image, [128,128])
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image


def random_rotation(image):
    k = tf.random_uniform(shape=[], minval=0, maxval=3, dtype=tf.int32)

    return tf.image.rot90(image, k)


def create_dataset(path, batch_size, img_height, img_width, channels, num_epochs):
    """
    This function does blah blah.
    """
    
    convert_tiff_to_jpeg(path)
    filenames = glob(os.path.join(path, "*.jpeg"))
    filenames.extend(glob(os.path.join(path, "*.jpg")))
    dataset = (tf.data.Dataset.from_tensor_slices((filenames))
                .repeat(num_epochs)
                .shuffle(buffer_size=len(filenames))
                .map(lambda filename: read_image(
                    filename, channels), num_parallel_calls=NUM_THREADS)
                
                .batch(batch_size)
                .prefetch(1))

    return dataset, len(filenames)


def extract_patches(image, patch_size):
    """extract square patches of 'patch_size' from the image"""

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
    This function does blah blah.
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
