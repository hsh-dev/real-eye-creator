import cv2
import os
import glob
import numpy as np
from dataset.data_loader import DataLoader

class Vw_Loader(DataLoader):
    def __init__(self, config):
        super().__init__(config)
        self.name = "300vw_blink"
        if config.get("300vw_blink_path") is not None:            
            self.dataset_path = config['300vw_blink_path']
        if config.get("test_300vw_blink_path") is not None:
            self.testset_path = config["test_300vw_blink_path"]
        self.input_size = config['input_size']

    def load_300vw(self, dataset_path, label):
        x = []
        y = []
        files = glob.glob(dataset_path+'/*')
        for image_path in files:
            img = cv2.imread(str(image_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.input_size, cv2.INTER_CUBIC)
            x.append(img)
            y.append(label)
        return np.array(x), y

    def load_300vw_blink(self):
        xs = []
        ys = []
        closed_path = os.path.join('../../300vw_blink', 'closed')
        opened_path = os.path.join('../../300vw_blink', 'opened')
        for i in range(1):
            c_x, c_y = self.load_dataset_300vw(closed_path, 1.0)
            xs.append(c_x)
            ys += c_y
            self.closed += len(c_y)
            o_x, o_y = self.load_dataset_300vw(opened_path, 0.0)
            xs.append(o_x)
            ys += o_y
            self.closed += len(o_y)
        subject = {'x': np.concatenate(xs), 'y': ys}
        return subject
