import cv2
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
import random
# from transforms import train_transform

from dataset.unity_loader import Unity_Loader
from dataset.rt_loader import Rt_Loader
from dataset.golflab_loader import Golflab_Loader

import albumentations as A

#(36, 60, 3)

train_transform = A.Compose([
    A.ISONoise(always_apply=False, p=0.3, intensity=(0.3, 0.55), color_shift=(0.2, 0.4)),
    A.Blur(always_apply=False, p=0.5, blur_limit=(3, 9)),
    A.ImageCompression(always_apply=False, p=1.0, quality_lower=52,quality_upper=100, compression_type=0),
    A.RandomBrightnessContrast(always_apply=False, p=0.5, brightness_limit=(-0.8, 0.2), 
                contrast_limit=(-0.8, 0.2), brightness_by_max=True),
    A.ShiftScaleRotate(always_apply=False, p=0.5, shift_limit=(-0.1, 0.1), scale_limit=(-0.1, 0.1))
])

# train_transform = A.Compose([
#     A.Blur(always_apply=False, p=0.5, blur_limit=(3, 9)),
#     A.Downscale(always_apply=False, p=0.1, scale_min=0.7,scale_max=0.99, interpolation=0),
#     A.ElasticTransform(always_apply=False, p=0.5, alpha=0.0, sigma=0.0, alpha_affine=35.5,
#                 interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None, approximate=False),
#     A.HueSaturationValue(always_apply=False, p=0.5, hue_shift_limit=(-20, 20),
#                 sat_shift_limit=(-30, 30), val_shift_limit=(-20, 20)),
#     A.ImageCompression(always_apply=False, p=1.0, quality_lower=52,quality_upper=100, compression_type=0),
#     A.JpegCompression(always_apply=False, p=1.0, quality_lower=49, quality_upper=100),
#     A.MotionBlur(always_apply=False, p=1.0, blur_limit=(3, 50)),
#     A.RGBShift(always_apply=False, p=1.0, r_shift_limit=(-10, 10),
#                 g_shift_limit=(-10, 10), b_shift_limit=(-10, 10)),
#     A.RandomBrightnessContrast(always_apply=False, p=1.0, brightness_limit=(-0.3, 0.3), 
#                 contrast_limit=(-0.2, 0.2), brightness_by_max=True),
#     A.RandomFog(always_apply=False, p=1.0, fog_coef_lower=0.0,
#                 fog_coef_upper=0.2, alpha_coef=0.13),
#     A.ShiftScaleRotate(always_apply=False, p=1.0, shift_limit=(-0.1, 0.1), scale_limit=(-0.1, 0.1),
#                 rotate_limit=(-30, 30), interpolation=0, border_mode=1, value=(0, 0, 0), mask_value=None),
#     A.ISONoise(always_apply=False, p=0.5, intensity=(0.3, 0.55), color_shift=(0.2, 0.4))
# ])

def aug_fn(image):
    data = {"image": image}
    aug_data = train_transform(**data)
    aug_img = aug_data["image"]
    return aug_img

@tf.function(input_signature=[tf.TensorSpec((36, 60, 3), tf.uint8)])
def _add_random_noise_each(image):
    aug_img = tf.numpy_function(func=aug_fn, inp=[image], Tout=tf.uint8)
    return aug_img


def image_normalize(image):
    image_norm = cv2.normalize(image, None, 50, 200, cv2.NORM_MINMAX)
    
    save_image = cv2.cvtColor(image_norm, cv2.COLOR_RGB2BGR)
    # cv2.imwrite('./output/save_image.jpg', save_image)
    
    return image_norm

# def load_one_flipped_pair(l_path, r_path, size):
#     l_img = read_rgb_image(l_path, size, flip=False)
#     r_img = read_rgb_image(r_path, size, flip=True)
#     return l_img, r_img


class DatasetManager(object):
    def __init__(self, config, finetune = False, dataset_initialize = True):
        '''
        dataset_size (None/int) : use whole dataset / use partial dataset
        '''
        self.config = config
        self.input_size = config['input_size']
        
        # self.subjects = {}
        self.train_set = {}
        self.valid_set = {}
        self.test_set = {}
        
        self.finetune = finetune
        self.is_partial = False
        self.size = None

        self.unity_loader = Unity_Loader(self.config)
        self.rt_loader = Rt_Loader(self.config)
        self.golflab_loader = Golflab_Loader(self.config)
        
        if dataset_initialize:
            if not self.finetune:            
                self.dataset_initialize()
            else:
                self.golflab_finetuning_initialize()

    def dataset_initialize(self):
        '''
        Initializing dataset, make train, valid, test set
        '''
        train_loader_list = []
        if "rt_bene" in self.config["train_dataset_list"]:
            train_loader_list.append(self.rt_loader)
        if "golflab" in self.config["train_dataset_list"]:
            train_loader_list.append(self.golflab_loader)
        if "unity_eyes" in self.config["train_dataset_list"]:
            train_loader_list.append(self.unity_loader)

        valid_loader_list = []
        if "rt_bene" in self.config["valid_dataset_list"]:
            valid_loader_list.append(self.rt_loader)
        if "golflab" in self.config["valid_dataset_list"]:
            valid_loader_list.append(self.golflab_loader)
        if "unity_eyes" in self.config["valid_dataset_list"]:
            valid_loader_list.append(self.unity_loader)


        test_loader_list = []
        if "rt_bene" in self.config["test_dataset_list"]:
            test_loader_list.append(self.rt_loader)
        if "golflab" in self.config["test_dataset_list"]:
            test_loader_list.append(self.golflab_loader)
        if "unity_eyes" in  self.config["test_dataset_list"]:
            test_loader_list.append(self.unity_loader)

        
        train_image, train_domain = self.load_data(loader_list = train_loader_list, type = 'train')
        valid_image, valid_domain = self.load_data(loader_list = valid_loader_list, type = 'valid')
        test_image, test_domain = self.load_data(loader_list = test_loader_list, type = 'test')
        
        # train_open_length = int(len(image["opened"])*0.8)
        # train_close_length = int(len(image["closed"])*0.8)
        
        # train_image = {}
        # valid_image = {}
        # train_domain = {}
        # valid_domain = {}
        
        # train_image["opened"] = image["opened"][:train_open_length]
        # train_domain["opened"] = domain["opened"][:train_open_length]
        # valid_image["opened"] = image["opened"][train_open_length:]
        # valid_domain["opened"] = domain["opened"][train_open_length:]
                
        # train_image["closed"] = image["closed"][:train_close_length]
        # train_domain["closed"] = domain["closed"][:train_close_length]
        # valid_image["closed"] = image["closed"][train_close_length:]
        # valid_domain["closed"] = domain["closed"][train_close_length:]

        self.concatenate_data(train_image, train_domain, 'train')
        self.concatenate_data(valid_image, valid_domain, 'valid')
        self.concatenate_data(test_image, test_domain, 'test')


    def golflab_finetuning_initialize(self):
        '''
        Initializing Golflab Finetuning
        '''
        loader_list = [
            self.golflab_loader
        ]

        train_image, train_domain = self.load_data(loader_list = loader_list, type = "train")
        valid_image, valid_domain = self.load_data(loader_list = loader_list, type = "valid")
        test_image, test_domain = self.load_data(loader_list = loader_list, type = "test")

        self.concatenate_data(train_image, train_domain, 'train')
        self.concatenate_data(valid_image, valid_domain, 'valid')
        self.concatenate_data(test_image, test_domain, 'test')

    
    ''' Load Function '''
    def load_data(self, loader_list, type):
        '''
        Load data from Loader List
        '''
        image_data_list = {}
        image_data_list["opened"] = []
        image_data_list["closed"] = []
        image_data_list["uncertain"] = []
        domain_data_list = {}
        domain_data_list["opened"] = []
        domain_data_list["closed"] = []

        for loader in loader_list:
            image_data_list, domain_data_list = self._load_data(image_data_list, domain_data_list, loader, type)
        
        data_type = type
        image_data_list["opened"], domain_data_list["opened"] = shuffle(image_data_list["opened"], domain_data_list["opened"], random_state=20)
        image_data_list["closed"], domain_data_list["closed"] = shuffle(image_data_list["closed"], domain_data_list["closed"], random_state=20)
        
        print("-------------------------------------")
        print("----  LOAD ALL {} DATA!!  ------".format(data_type))
        print("[*] Total Opened Size : {}".format(len(image_data_list["opened"])))
        print("[*] Total Closed Size : {}".format(len(image_data_list["closed"])))
        print("[*] Total Uncertain Size : {}".format(len(image_data_list["uncertain"])))
        print("-------------------------------------\n")

        return image_data_list, domain_data_list

    def _load_data(self, image_data, domain_data, loader, type):
        opened_set, closed_set, uncertain_set, domain_set = loader.get_data(type)
        image_data["opened"].extend(opened_set)
        image_data["closed"].extend(closed_set)
        image_data["uncertain"].extend(uncertain_set)
        domain_data["opened"].extend(domain_set["opened"])
        domain_data["closed"].extend(domain_set["closed"])
        return image_data, domain_data
    
    
    def concatenate_data(self, image, domain, type):
        '''
        Initialize self train/valid test set by concatenating parameter
        '''
        total_image = []
        total_domain = []
        total_y = []
        
        total_image.extend(image["opened"])
        total_image.extend(image["closed"])
        total_domain.extend(domain["opened"])
        total_domain.extend(domain["closed"])
                
        for i in range(len(image["opened"])):
            total_y.append(0.0)
        for i in range(len(image["closed"])):
            total_y.append(1.0)
        
        zip_set = list(zip(total_image, total_domain, total_y))
        random.shuffle(zip_set)
        total_image, total_domain, total_y = zip(*zip_set)
        total_image = list(total_image)
        total_domain = list(total_domain)
        total_y = list(total_y)

        if type == "test":
            self.test_set['x'] = total_image
            self.test_set['y'] = total_y
            self.test_set['d'] = total_domain
        elif type == "valid":
            self.valid_set['x'] = total_image
            self.valid_set['y'] = total_y
            self.valid_set['d'] = total_domain
        elif type == "train":
            self.train_set['x'] = total_image
            self.train_set['y'] = total_y
            self.train_set['d'] = total_domain
            
        print("-------------------------------------")
        print("[*] {} set : {}".format(type, len(total_y)))
        print("[->] {} opened : {} | {} closed : {}".format(type, len(image["opened"]), type, len(image["closed"])))
        print("-------------------------------------")



    def get_data(self, dataset, training, batch_size, finetuning = False):
        all_x = np.array(dataset['x'])
        # if finetuning:
        #     all_x = np.array(
        #         list(map(
        #             lambda x : image_normalize(x), dataset['x']
        #         )))
        all_y = np.array(dataset['y'], dtype=np.float32)
        all_d = np.array(dataset['d'], dtype=np.float32)
        
        
        # all_x, all_y = shuffle(all_x, all_y)
        dataset = tf.data.Dataset.from_tensor_slices((all_x, all_y, all_d))
        
        if training:
            dataset = dataset.map(lambda x, y, d: (tf.image.convert_image_dtype(_add_random_noise_each(x), dtype=tf.float32), y, d))\
            .batch(batch_size).shuffle(512).prefetch(tf.data.experimental.AUTOTUNE)
        else:
            dataset = dataset.map(lambda x, y, d: (tf.image.convert_image_dtype(x, dtype=tf.float32), y, d)).batch(batch_size)
        
        return dataset


    def read_rgb_image(self, img_path, flip=False):
        assert type(self.input_size) is tuple, "size parameter must be a tuple"
        assert ('jpg' in img_path) or ('png' in img_path), "this is not image path"
        
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print("ERROR: can't read " + img_path)
            return None
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if flip:
                img = cv2.flip(img, 1)
        
            img = cv2.resize(img, self.input_size, cv2.INTER_CUBIC)
            return img


    '''Data API functions'''
    def get_training_data(self, batch_size):
        subject_list = self.train_set.keys()
        if self.finetune:
            return self.get_data(self.train_set, False, batch_size, True)
        else:
            return self.get_data(self.train_set, True, batch_size)
    
    def get_validation_data(self, batch_size):
        subject_list = self.valid_set.keys()
        if self.finetune:
            return self.get_data(self.valid_set, False, batch_size, True)
        else:
            return self.get_data(self.valid_set, False, batch_size)

    
    def get_test_data(self, batch_size):
        subject_list = self.test_set.keys()
        if self.finetune:
            return self.get_data(self.test_set, False, batch_size, True)
        else:
            return self.get_data(self.test_set, False, batch_size)
