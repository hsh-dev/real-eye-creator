from dataset.data_loader import DataLoader

class Unity_Loader(DataLoader):
    def __init__(self, config):
        super().__init__(config)
        self.name = "unity_eyes"
        if config.get("unity_eyes_path") is not None:
            self.dataset_path = config['unity_eyes_path']
        if config.get("test_unity_eyes_path") is not None:
            self.testset_path = config["test_unity_eyes_path"]
        self.input_size = config['input_size']

