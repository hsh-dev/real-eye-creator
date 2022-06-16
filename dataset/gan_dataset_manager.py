from xml import dom

from dataset.dataset_manager import DatasetManager
from random import shuffle

import numpy as np
import tensorflow as tf

class Gan_DatasetManager(DatasetManager):
    def __init__(self, config, finetune=False):
        super().__init__(config, finetune, False)
        self.batch_size = config['batch_size']
        
        self.train_set_list = [
            'unity', 'rt_bene', 'golflab'
        ]
        self.data_set = {}
        self.gan_dataset_initialize()
        
    def gan_dataset_initialize(self):
        '''
        Initializing dataset, make train, valid, test set \n
        Split test set and train/validation set
        '''
        
        for domain in self.train_set_list:
            loader = None
            if domain == "unity":
                loader = self.unity_loader
            elif domain == "rt_bene":
                loader = self.rt_loader
            elif domain == "golflab":
                loader = self.golflab_loader
            # self.vw_loader
            # self.golflab_loader

            loader_list = []
            loader_list.append(loader)
            opened_data, closed_data, uncertain_data = [], [], []
            if domain == "golflab":
                opened_data, closed_data, uncertain_data = self.load_data(loader_list, 'train')
            else:
                opened_data, closed_data, uncertain_data = self.load_data(loader_list, 'train')
        
            self.concatenate_data(opened_data, closed_data, [], domain)
        
            print("-------------------------------------") 
            print("[*] Domain : {}".format(domain))
            print("[*] Data Number : {}".format(len(self.data_set[domain]['image'])))
            print("-------------------------------------")
    
    
    def concatenate_data(self, opened_data, closed_data, uncertain_data, domain):
        '''
        Initialize self train/valid test set by concatenating parameter
        '''
        total_data = []
        total_data.extend(opened_data)
        total_data.extend(closed_data)
        
        total_domain = []
        domain_vector = None
        if domain == 'unity':
            domain_vector = [1.0, 0.0, 0.0]
        elif domain == 'rt_bene':
            domain_vector = [0.0, 1.0, 0.0]
        elif domain == 'golflab':
            domain_vector = [0.0, 0.0, 1.0]
        
        for i in range(len(total_data)):
            total_domain.append(domain_vector)
        
        total_state = []
        for i in range(len(opened_data)):
            total_state.append(0.0)
        for i in range(len(closed_data)):
            total_state.append(1.0)
        
        total_data, total_domain, total_state = self.shuffle_list(total_data, total_domain, total_state)
        
        self.data_set[domain] = {}
        self.data_set[domain]['image'] = total_data
        self.data_set[domain]['domain'] = total_domain
        self.data_set[domain]['state'] = total_state

            
    def get_data(self, batch_size, training = None):
        all_real_img = np.array(self.data_set[self.train_set_list[0]]['image'])
        all_real_d = np.array(self.data_set[self.train_set_list[0]]['domain'])
        all_real_s = np.array(self.data_set[self.train_set_list[0]]['state'])
        
        for i in range(1, len(self.train_set_list)):
            domain = self.train_set_list[i]
            all_real_img = np.concatenate((all_real_img, self.data_set[domain]['image']), axis=0)
            all_real_d = np.concatenate((all_real_d, self.data_set[domain]['domain']), axis=0)
            all_real_s = np.concatenate((all_real_s, self.data_set[domain]['state']), axis=0)
            
        # all_fake, all_real = shuffle(all_fake, all_real)
        dataset = tf.data.Dataset.from_tensor_slices((
            all_real_img, all_real_d, all_real_s
        ))
        
        dataset = dataset.map(
            lambda img, d, s : (
                tf.image.convert_image_dtype(
                    img, dtype=tf.float32),
                tf.cast(d, dtype=tf.float32),
                tf.cast(s, dtype=tf.float32)
        )).batch(batch_size).shuffle(buffer_size=len(all_real_img)).prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

    def shuffle_list(self, *ls):
        l =list(zip(*ls))

        shuffle(l)
        return zip(*l)