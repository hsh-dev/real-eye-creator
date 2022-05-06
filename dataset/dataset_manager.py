import csv
import os
import random
from this import d
import cv2
import glob
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
# from transforms import train_transform
from tqdm import tqdm

from dataset.unity_loader import Unity_Loader
from dataset.vw_loader import Vw_Loader
from dataset.rt_loader import Rt_Loader

import albumentations as A

#(36, 60, 3)

train_transform = A.Compose([
    A.Blur(always_apply=False, p=0.5, blur_limit=(3, 9)),
    A.Downscale(always_apply=False, p=0.1, scale_min=0.7,
                scale_max=0.99, interpolation=0),
    A.ElasticTransform(always_apply=False, p=0.5, alpha=0.0, sigma=0.0, alpha_affine=35.5,
                       interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None, approximate=False),
    A.HueSaturationValue(always_apply=False, p=0.5, hue_shift_limit=(-20, 20),
                         sat_shift_limit=(-30, 30), val_shift_limit=(-20, 20)),
    A.ImageCompression(always_apply=False, p=1.0, quality_lower=52,
                       quality_upper=100, compression_type=0),
    A.JpegCompression(always_apply=False, p=1.0,
                      quality_lower=49, quality_upper=100),
    A.MotionBlur(always_apply=False, p=1.0, blur_limit=(3, 50)),
    A.RGBShift(always_apply=False, p=1.0, r_shift_limit=(-10, 10),
               g_shift_limit=(-10, 10), b_shift_limit=(-10, 10)),
    A.RandomBrightnessContrast(always_apply=False, p=1.0, brightness_limit=(-0.3, 0.3),
                               contrast_limit=(-0.2, 0.2), brightness_by_max=True),
    A.RandomFog(always_apply=False, p=1.0, fog_coef_lower=0.0,
                fog_coef_upper=0.2, alpha_coef=0.13),
    A.ShiftScaleRotate(always_apply=False, p=1.0, shift_limit=(-0.1, 0.1), scale_limit=(-0.1, 0.1),
                       rotate_limit=(-30, 30), interpolation=0, border_mode=1, value=(0, 0, 0), mask_value=None)

])


def aug_fn(image):
    data = {"image": image}
    aug_data = train_transform(**data)
    aug_img = aug_data["image"]
    return aug_img


@tf.function(input_signature=[tf.TensorSpec((36, 60, 3), tf.uint8)])
def _add_random_noise_each(image):
    aug_img = tf.numpy_function(func=aug_fn, inp=[image], Tout=tf.uint8)
    return aug_img


class DatasetManager(object):
    def __init__(self, config, test_split = False, dataset_size=None):
        '''
        test_split (T/F) : make test set from dataset / no test set \n
        dataset_size (None/int) : use whole dataset / use partial dataset
        '''
        self.config = config
        self.input_size = config['input_size']

        self.train_set = {}     # real image(Rt_BENE) /  fake image(Unity Eyes)
        self.valid_set = {}     # fake validation image (Unity Eyes)

        self.is_partial = False
        self.size = None

        if dataset_size is not None:
            self.is_partial = True
            self.size = dataset_size

        self.dataset_initialize()
        # self.unityeye_test_dataset_initialize()


    def dataset_initialize(self):
        '''
        Initializing dataset, make train, valid, test set \n
        Split test set and train/validation set
        '''

        vw_loader = Vw_Loader(self.config)
        unity_loader = Unity_Loader(self.config)
        rt_loader = Rt_Loader(self.config)

        real_opened_set = []

        real_open, real_closed = vw_loader.get_data_path()
        real_opened_set.extend(real_open)
        real_open, real_closed = rt_loader.get_data_path()
        real_opened_set.extend(real_open)

        fake_opened_set, fake_closed_set = unity_loader.get_data_path()

        fake_opened_set = shuffle(fake_opened_set)
        real_opened_set = shuffle(real_opened_set)
        

        valid_length = int(min(len(fake_opened_set), len(real_opened_set)) * 0.2)
        self.valid_set['fake'] = fake_opened_set[-valid_length:]

        train_length = len(fake_opened_set) - valid_length
        real_length = len(real_opened_set)

        if train_length > real_length:
            self.train_set['real'] = real_opened_set
            self.train_set['fake'] = fake_opened_set[:real_length]
        else:
            self.train_set['real'] = real_opened_set[:train_length]
            self.train_set['fake'] = fake_opened_set[:train_length]


        print("[*] Train Real Set : {}".format(len(self.train_set['real'])))
        print("[*] Train Fake Set : {}".format(len(self.train_set['fake'])))
        print("[*] Valid Set : {}".format(len(self.valid_set['fake'])))
        print("-------------------------------------")


    def load_data_path(self):
        '''
        Load path from 300vw, unity eyes, rt_bene dataset
        '''
        vw_loader = Vw_Loader(self.config)
        unity_loader = Unity_Loader(self.config)
        rt_loader = Rt_Loader(self.config)
        
        opened_path_list = []
        closed_path_list = []

        opened_set, closed_set = vw_loader.get_data_path()
        opened_path_list.extend(opened_set)
        closed_path_list.extend(closed_set)

        opened_set, closed_set = unity_loader.get_data_path()
        opened_path_list.extend(opened_set)
        closed_path_list.extend(closed_set)
        
        opened_set, closed_set = rt_loader.get_data_path()
        opened_path_list.extend(opened_set)
        closed_path_list.extend(closed_set)

        print("-------------------------------------")
        print("[*] Total Opened Size : {}".format(len(opened_path_list)))
        print("[*] Total Closed Size : {}".format(len(closed_path_list)))
        print("-------------------------------------")

        return opened_path_list, closed_path_list


    def get_data(self, dataset, training, batch_size):
        all_fake = np.array(list(map(lambda x: self.read_rgb_image(x), dataset['fake'])))
        all_real = np.array(list(map(lambda x : self.read_rgb_image(x), dataset['real'])))

        all_fake, all_real = shuffle(all_fake, all_real)
        dataset = tf.data.Dataset.from_tensor_slices((all_fake, all_real))


        dataset = dataset.map(lambda x,y : (
            tf.image.convert_image_dtype(
                x, dtype=tf.float32),
            tf.image.convert_image_dtype(
                y, dtype=tf.float32)
        )).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        return dataset
        

    def read_rgb_image(self, img_path, flip=False):
        assert type(self.input_size) is tuple, "size parameter must be a tuple"
        assert ('jpg' in img_path) or ('png' in img_path), "this is not image path"
        
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print("ERROR: can't read " + img_path)
            return None
        else:
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if flip:
                img = cv2.flip(img, 1)
        
            img = cv2.resize(img, self.input_size, cv2.INTER_CUBIC)
            return img


    '''Data API functions'''
    def get_training_data(self, batch_size):
        subject_list = self.train_set.keys()
        return self.get_data(self.train_set, False, batch_size)
    
    def get_validation_data(self, batch_size):
        subject_list = self.valid_set.keys()
        return self.get_data(self.valid_set, False, batch_size)
    
    def get_test_data(self, batch_size):
        subject_list = self.test_set.keys()
        return self.get_data(self.test_set, False, batch_size)

    def get_one_training_data(self):
        return self.get_data(self.train_set, False, 1)
