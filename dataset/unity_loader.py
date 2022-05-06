from dataset.data_loader import DataLoader

class Unity_Loader(DataLoader):
    def __init__(self, config):
        super().__init__(config)
        self.name = "unity_eyes"
        self.dataset_path = config['unity_eyes_path']
        self.input_size = config['input_size']

    def get_data_path(self):
        opened_path, closed_path = self._get_data_path(
            self.name, self.dataset_path)
        return opened_path, closed_path
