from dataset.data_loader import DataLoader
import os
import pandas as pd
import numpy as np
from config_file.domain_map import DOMAIN_ID


class Golflab_Loader(DataLoader):
    def __init__(self, config):
        super().__init__(config)
        self.name = "golflab"

        if config.get("golflab_path") is not None:
            self.dataset_path = config["golflab_path"]
            self.testset_path = config["golflab_path"]
        self.input_size = config['input_size']

        self.train_subject_list = config['golflab_train_subject']
        self.valid_subject_list = config['golflab_valid_subject']
        self.test_subject_list = config['golflab_test_subject']
    
    
    # override _get_data_from_path function
    def _get_data_from_path(self, name, dataset_path, type):
        opened_path_list = []
        closed_path_list = []
        uncertain_path_list = []
        
        if not self._is_data_exist(name, dataset_path):
            return np.array(opened_path_list), np.array(closed_path_list), np.array(uncertain_path_list)
        else:
            print("Collecting {} dataset...".format(name))

            opened_path_list, closed_path_list, uncertain_path_list = self.read_csv(dataset_path, type)
            
            opened_data = np.array(list(map(
                lambda x : self.read_rgb_image(x), opened_path_list
                )))
            closed_data = np.array(list(map(
                lambda x : self.read_rgb_image(x), closed_path_list
                )))
            uncertain_data = np.array(list(map(
                lambda x : self.read_rgb_image(x), uncertain_path_list
                )))

            subject_list = []
            if type == "train":
                subject_list = self.train_subject_list
            elif type == "valid":
                subject_list = self.valid_subject_list
            elif type == "test":
                subject_list = self.test_subject_list
            print("#### Golflab Load Info ####")
            print("{} set from ---> {}".format(type, subject_list))
            print("[*] Opened Dataset Size : {}".format(len(opened_data)))
            print("[*] Closed Dataset Size : {}".format(len(closed_data)))
            print("[*] Uncertain Dataset Size : {}".format(len(uncertain_data)))            
            print("\n")

            return opened_data, closed_data, uncertain_data

    def _get_csv_list(self, dataset_path):
        csv_path = os.path.join(dataset_path, "csv_file")
        file_list = os.listdir(csv_path)
        csv_file_list = []

        for file in file_list:
            if 'csv' in file:
                csv_file_list.append(file)  
        return csv_file_list


    def read_csv(self, dataset_path, type):
        csv_file_list = self._get_csv_list(dataset_path)
        opened_path_list = []
        closed_path_list = []
        uncertain_path_list = []
        
        for csv_file_name in csv_file_list:
            subject_name = csv_file_name.split('_')[0] + '_' + csv_file_name.split('_')[1]
            subject_id = subject_name.split('_')[0]
            subject_path = os.path.join(dataset_path, subject_name)
            
            check_flag = self.check_subject(type,subject_id)
            if not check_flag:
                continue
            
            csv_file_path = os.path.join(os.path.join(dataset_path, "csv_file"), csv_file_name)
            
            df = pd.read_csv(csv_file_path, index_col=0)
            image_column = df["image"].tolist() 
            
            for idx, image in enumerate(image_column):
                img_path = os.path.join(subject_path, image)
                if df.at[idx, "label"] == 2:
                    uncertain_path_list.append(img_path)
                elif df.at[idx, "label"] == 1:
                    closed_path_list.append(img_path)
                elif df.at[idx, "label"] == 0:
                    opened_path_list.append(img_path)
                    
        return opened_path_list, closed_path_list, uncertain_path_list
    

    def check_subject(self, type, subject_id):
        subject_id = int(subject_id)
        if type == "test" and (subject_id in self.test_subject_list):
            return True
        if type == "valid" and (subject_id in self.valid_subject_list):
            return True
        if type == "train" and (subject_id in self.train_subject_list):
            return True
        
        return False