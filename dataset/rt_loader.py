import random
from dataset.data_loader import DataLoader
import os
import csv
from config_file.domain_map import DOMAIN_ID
from config_file.random_seed import SEED
import numpy as np

class Rt_Loader(DataLoader):
    def __init__(self, config):
        super().__init__(config)
        self.name = "rt_bene"

        if config.get("rt_bene_path") is not None:
            self.dataset_path = config['rt_bene_path']
            self.testset_path = config['rt_bene_path']

        self.input_size = config['input_size']
        self.train_subject_list = config['rt_bene_train_subject']
        self.valid_subject_list = config['rt_bene_valid_subject']
        self.test_subject_list = config['rt_bene_test_subject']


    def _get_data_from_path(self, name, dataset_path, type):
        opened_path_list = []
        closed_path_list = []
        uncertain_path_list = []

        if not self._is_data_exist(name, dataset_path):
            return np.array(opened_path_list), np.array(closed_path_list), np.array(uncertain_path_list)
        else:
            opened_path, closed_path, uncertain_path = self.get_raw_data_path()

            left_opened_path = []
            left_closed_path = []
            left_uncertain_path = []
            right_opened_path = []
            right_closed_path = []
            right_uncertain_path = []

            self.subject_slice(opened_path, left_opened_path, right_opened_path, type)
            self.subject_slice(closed_path, left_closed_path, right_closed_path, type)
            self.subject_slice(uncertain_path, left_uncertain_path, right_uncertain_path, type)

            opened_data = self.concate_data(left_opened_path, right_opened_path)
            closed_data = self.concate_data(left_closed_path, right_closed_path)
            uncertain_data = self.concate_data(left_uncertain_path, right_uncertain_path)

            if (type == "test" and self.config['test_rt_bene_resize']) or ((type == "train" or type == "valid") and self.config['train_rt_bene_resize']):
                resized_length = min(len(closed_data) * self.config["rt_bene_resize_ratio"], len(opened_data))
                opened_data = opened_data[:resized_length]
                closed_data = closed_data[:resized_length]


            subject_list = []
            if type == "train":
                subject_list = self.train_subject_list
            elif type == "valid":
                subject_list = self.valid_subject_list
            elif type == "test":
                subject_list = self.test_subject_list
            print("#### RT_BENE Load Info ####")
            print("{} set from ---> {}".format(type, subject_list))
            print("[*] Opened Dataset Size : {}".format(len(opened_data)))
            print("[*] Closed Dataset Size : {}".format(len(closed_data)))
            print("[*] Uncertain Dataset Size : {}".format(len(uncertain_data)))

            return opened_data, closed_data, uncertain_data

    def concate_data(self, left_path, right_path):
        left_opened_list = list(map(lambda x : self.read_rgb_image(x), left_path))
        right_opened_list = list(map(lambda x : self.read_rgb_image(x, True), right_path))
        
        total_list = []
        total_list.extend(left_opened_list)
        total_list.extend(right_opened_list)

        random.Random(SEED["random"]).shuffle(total_list)

        total_data = np.array(total_list)
        return total_data

    def subject_slice(self, target_path, left_path, right_path, type):
        for path in target_path:
            subject_name = path.split('/')[-4].split('_')[0][2:4]
            rl_type = path.split('/')[-1].split('_')[0]
            
            is_valid = self.check_subject(type, subject_name)
            if is_valid and rl_type == "left":
                left_path.append(path)
            elif is_valid and rl_type == "right":
                right_path.append(path)

    def check_subject(self, type, subject_id):
        subject_num = None
        if subject_id[0] == "0":
            subject_num = int(subject_id[1])
        else:
            subject_num = int(subject_id)
        
        if type == "train":
            if subject_num in self.train_subject_list:
                return True
        elif type == "valid":
            if subject_num in self.valid_subject_list:
                return True
        elif type == "test":
            if subject_num in self.test_subject_list:
                return True
        return False



    def get_raw_data_path(self):
        '''
        Raw RT_BENE
        '''        
        opened_path = []
        closed_path = []
        uncertain_path = []

        print("Collecting {} dataset...".format(self.name))
        opened_path, closed_path, uncertain_path = self.read_csv()

        print("[*] Opened Dataset Size : {}".format(len(opened_path)))
        print("[*] Closed Dataset Size : {}".format(len(closed_path)))
        print("[*] Uncertain Dataset Size : {}".format(len(uncertain_path)))
        print("\n")
        return opened_path, closed_path, uncertain_path
    
    def _get_csv_list(self):
        file_list = os.listdir(self.dataset_path)
        csv_file_list = []

        for file in file_list:
            if 'csv' in file:
                if not 'rt_bene' in file:
                    csv_file_list.append(file)  
        return csv_file_list

    def read_csv(self):
        csv_file_list = self._get_csv_list()
        opened_path_list = []
        closed_path_list = []
        uncertain_path_list = []

        for csv_file in csv_file_list:
            subject_path = os.path.join(self.dataset_path, csv_file.split('_')[0] + '_noglasses')
            left_subject_path = os.path.join(subject_path, 'natural/left')
            right_subject_path = os.path.join(subject_path, 'natural/right')

            with open(os.path.join(self.dataset_path,csv_file)) as csvfile:
                csv_rows = csv.reader(csvfile)

                for row in csv_rows:
                    left_img_name = row[0]
                    right_img_name = left_img_name.replace('left', 'right')      
                    img_label = float(row[1])
                    left_img_path = os.path.join(left_subject_path, left_img_name)
                    rigth_img_path = os.path.join(right_subject_path, right_img_name)

                    if img_label == 0.5:
                        uncertain_path_list.append(left_img_path)
                        uncertain_path_list.append(rigth_img_path)
                    if img_label == 1.0:
                        closed_path_list.append(left_img_path)
                        closed_path_list.append(rigth_img_path)
                    elif img_label == 0.0:
                        opened_path_list.append(left_img_path)
                        opened_path_list.append(rigth_img_path)

        return opened_path_list, closed_path_list, uncertain_path_list

