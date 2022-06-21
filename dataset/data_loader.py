import cv2
import os
import glob
import numpy as np
from sklearn.utils import shuffle
from config_file.domain_map import DOMAIN_ID

class DataLoader():
    def __init__(self, config):
        self.config = config
        self.name = None
        self.dataset_path = None
        self.testset_path = None
    
    def get_data(self, type):
        opened_data = []
        closed_data = []
        uncertain_data = []
        if (type == "train" or type == "valid") and (self.dataset_path is not None):
            opened_data, closed_data, uncertain_data = self._get_data_from_path(
            self.name, self.dataset_path, type)
        if type == "test" and (self.testset_path is not None):
            opened_data, closed_data, uncertain_data = self._get_data_from_path(
                self.name, self.testset_path, type)
        
        target_domain = [0]*3
        target_domain_id = DOMAIN_ID[self.name]
        target_domain[target_domain_id] = 1
        domain_data = {}
        domain_data["opened"] = []
        domain_data["closed"] = []
        for i in range(len(opened_data)):
            domain_data["opened"].append(target_domain)
        for i in range(len(closed_data)):
            domain_data["closed"].append(target_domain)
                    
        return opened_data, closed_data, uncertain_data, domain_data        
        
        
    def _get_data_from_path(self, name, dataset_path, type):
        opened_path_list = []
        closed_path_list = []
        uncertain_path_list = []
        
        test = False
        if type == "test":
            test = True

        if not self._is_data_exist(name, dataset_path):
            return np.array(opened_path_list), np.array(closed_path_list), np.array(uncertain_path_list)
        else:
            print("Collecting {} dataset...".format(name))

            opened_path = os.path.join(dataset_path, 'opened')
            closed_path = os.path.join(dataset_path, 'closed')
            uncertain_path = os.path.join(dataset_path, 'uncertain')
            
            for file in os.listdir(opened_path):
                if ('jpg' in file) or ('png' in file):
                    opened_path_list.append(os.path.join(opened_path, file))

            for file in os.listdir(closed_path):
                if ('jpg' in file) or ('png' in file):  
                    closed_path_list.append(os.path.join(closed_path, file))
                    
            for file in os.listdir(uncertain_path):
                if ('jpg' in file) or ('png' in file):  
                    uncertain_path_list.append(os.path.join(uncertain_path, file))
            
            opened_path_list = shuffle(opened_path_list, random_state=20)
            closed_path_list = shuffle(closed_path_list, random_state=20)
            
            opened_data = np.array(list(map(
                lambda x : self.read_rgb_image(x), opened_path_list
                )))
            closed_data = np.array(list(map(
                lambda x : self.read_rgb_image(x), closed_path_list
                )))
            uncertain_data = np.array(list(map(
                lambda x : self.read_rgb_image(x), uncertain_path_list
                )))
            
            if name == "unity_eyes":
                if (test and self.config['test_unity_eyes_resize']) or (not test and self.config['train_unity_eyes_resize']):
                    opened_data = opened_data[:10000]
                    closed_data = closed_data[:10000]
                    uncertain_data = uncertain_data[:10000]
                
            if name == "rt_bene":
                if (test and self.config['test_rt_bene_resize']) or (not test and self.config['train_rt_bene_resize']):
                    min_length = min(len(opened_data), len(closed_data))
                    opened_data = opened_data[:min_length]
                    closed_data = closed_data[:min_length]
            
            print("[*] Opened Dataset Size : {}".format(len(opened_data)))
            print("[*] Closed Dataset Size : {}".format(len(closed_data)))
            print("[*] Uncertain Dataset Size : {}".format(len(uncertain_data)))            
            print("\n")

            return opened_data, closed_data, uncertain_data
    
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
        
    def _is_data_exist(self, name, dataset_path):
        if not os.path.isdir(dataset_path):
            print("[!] {} dataset is not existing in".format(name))
            print(dataset_path)
            print("check the right path..." + '\n')
            return False

        if name == "unity_eyes":
            opened_path = os.path.join(dataset_path, 'opened')
            closed_path = os.path.join(dataset_path, 'closed')
            uncertain_path = os.path.join(dataset_path, 'uncertain')
            
            if not os.path.isdir(opened_path):
                print("[!] {} dataset opened path is not existing in".format(name))
                print(opened_path)
                print("check the right path..." + '\n')
                return False

            if not os.path.isdir(closed_path):
                print("[!] {} dataset closed path is not existing in".format(name))
                print(closed_path)
                print("check the right path..." + '\n')
                return False

            if not os.path.isdir(uncertain_path):
                print("[!] {} dataset uncertain path is not existing in".format(name))
                print(uncertain_path)
                print("check the right path..." + '\n')
                
            print("***********************************")
            print("[*] {} dataset found!".format(name))
            print("[*] Path in...")
            print("[*] " + dataset_path + '\n')

            return True
        else:
            print("***********************************")
            print("[*] {} dataset found!".format(name))
            print("[*] Path in...")
            print("[*] " + dataset_path + '\n')
            
            return True
