import cv2
import os
import glob
import numpy as np
from tqdm import tqdm

class DataLoader():
    def __init__(self, config):
        self.config = config
        
    def _get_data_path(self, name, dataset_path):
        opened_path_list = []
        closed_path_list = []

        if not self._is_data_exist(name, dataset_path):
            return opened_path_list, closed_path_list
        else:
            print("Collecting {} dataset...".format(name))

            opened_path = os.path.join(dataset_path, 'opened')
            closed_path = os.path.join(dataset_path, 'closed')
            
            ## Assume that 300vw dataset is preprocessed (normalized as left image)
            for file in tqdm(os.listdir(opened_path)):
                if ('jpg' in file) or ('png' in file):
                    opened_path_list.append(os.path.join(opened_path, file))

            for file in tqdm(os.listdir(closed_path)):
                if ('jpg' in file) or ('png' in file):  
                    closed_path_list.append(os.path.join(closed_path, file))

            print("[*] Opened Dataset Size : {}".format(len(opened_path_list)))
            print("[*] Closed Dataset Size : {}".format(len(closed_path_list)))
            print("\n")

            return opened_path_list, closed_path_list

    def _is_data_exist(self, name, dataset_path):
        if not os.path.isdir(dataset_path):
            print("[!] {} dataset is not existing in".format(name))
            print(dataset_path)
            print("check the right path..." + '\n')
            return False

        if name == "300vw_blink" or name == "unity_eyes":
            opened_path = os.path.join(dataset_path, 'opened')
            closed_path = os.path.join(dataset_path, 'closed')

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
            print("***********************************")
            print("[*] {} dataset found!".format(name))
            print("[*] Path in...")
            print("[*] " + dataset_path + '\n')

            return True
        else:
            return True
