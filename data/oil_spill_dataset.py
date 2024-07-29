import os
from glob import glob

from tensorflow import image as tf_image
from tensorflow import data as tf_data
from tensorflow import io as tf_io

IMAGE_SIZE = 256
BATCH_SIZE = 4
NUM_CLASSES = 5
NUM_INPUTS = 3
HOME_DIR = os.path.expanduser('~')
DATA_DIR = os.path.join(HOME_DIR, "data", "oil-spill-dataset_256")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")
print(DATA_DIR, VAL_DIR, TEST_DIR)
NUM_TRAIN_IMAGES = 1000
NUM_VAL_IMAGES = 50

train_images = sorted(glob(os.path.join(TRAIN_DIR, "images/*")))
train_masks = sorted(glob(os.path.join(TRAIN_DIR, "labels_1D/*")))
val_images = sorted(glob(os.path.join(VAL_DIR, "images/*")))
val_masks = sorted(glob(os.path.join(VAL_DIR, "labels_1D/*")))
print(len(train_images), len(train_masks), len(val_images), len(val_masks))

def read_image(image_path, mask=False):
    image = tf_io.read_file(image_path)
    if mask:
        image = tf_image.decode_png(image, channels=1)
        image.set_shape([None, None, 1])
        image = tf_image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
    else:
        image = tf_image.decode_jpeg(image, channels=NUM_INPUTS)
        image.set_shape([None, None, NUM_INPUTS])
        image = tf_image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
    return image


def load_data(image_list, mask_list):
    image = read_image(image_list)
    mask = read_image(mask_list, mask=True)
    return image, mask


def data_generator(image_list, mask_list):
    dataset = tf_data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf_data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset

train_dataset = data_generator(train_images, train_masks)
val_dataset = data_generator(val_images, val_masks)

