from dataset.data_loader import DataLoader
import os
import csv
from tqdm import tqdm 

class Golflab_Loader(DataLoader):
    def __init__(self, config):
        super().__init__(config)
        self.name = "golflab"
        if config.get("golflab_path") is not None:
            self.dataset_path = config["golflab_path"]
        if config.get("test_golflab_path") is not None:
            self.testset_path = config["test_golflab_path"]
        self.input_size = config['input_size']
