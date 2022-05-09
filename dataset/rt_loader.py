from dataset.data_loader import DataLoader
import os
import csv
from tqdm import tqdm 

class Rt_Loader(DataLoader):
    def __init__(self, config):
        super().__init__(config)
        self.name = "rt_bene"
        self.dataset_path = config['rt_bene_path']
        self.input_size = config['input_size']

    def get_data_path(self):
        is_exist = self._is_data_exist(self.name, self.dataset_path)
        
        opened_path = []
        closed_path = []

        if not is_exist:
            return opened_path, closed_path
        else:
            print("Collecting {} dataset...".format(self.name))
            opened_path, closed_path = self.read_csv()

            print("[*] Opened Dataset Size : {}".format(len(opened_path)))
            print("[*] Closed Dataset Size : {}".format(len(closed_path)))
            print("\n")
            return opened_path, closed_path
    
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

        for csv_file in csv_file_list:
            subject_path = os.path.join(self.dataset_path, csv_file.split('_')[0] + '_noglasses')
            subject_path = os.path.join(subject_path, 'natural/left')

            with open(os.path.join(self.dataset_path,csv_file)) as csvfile:
                csv_rows = csv.reader(csvfile)

                for row in tqdm(csv_rows):
                    img_name = row[0]
                    img_label = float(row[1])
                    img_path = os.path.join(subject_path, img_name)

                    if img_label == 0.5:
                        continue
                    if img_label == 1.0:
                        closed_path_list.append(img_path)
                    elif img_label == 0.0:
                        opened_path_list.append(img_path)

        return opened_path_list, closed_path_list

