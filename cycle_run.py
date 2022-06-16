import tensorflow as tf
import argparse
import os
import neptune.new as neptune

from utils.dir import check_make_dir
from dataset.dataset_manager import DatasetManager
# from trainer.trainer import Trainer
from trainer.cyclegan_trainer import Cycle_Trainer
from models.real_eye_gan import RE_Discriminator, RE_Generator, Generator, Discriminator
from config import CONFIG

if __name__ == "__main__":    
    tf.debugging.set_log_device_placement(False)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(
                logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save_path", default="./output")
    parser.add_argument("-n", "--experiment_name", default="gan_test")
    parser.add_argument(
        "--enable_log",
        help="Decide whether to upload log on neptune or not",
        action='store_true'
    )

    args = parser.parse_args()
    output_path = os.path.abspath(args.save_path)
    check_make_dir(output_path)
    CONFIG["save_path"] = os.path.join(args.save_path, args.experiment_name)
    check_make_dir(CONFIG["save_path"])

    neptune_callback = None
    if args.enable_log:
        neptune_callback = neptune.init(
            name=args.experiment_name,
            project="vcamp/jeff-flip21",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0YzkwOTk5Zi03ZGE1LTQ2MjQtYTkxNC0wZTNiN2I1Y2M5OTkifQ==",
            source_files=[
                "run.py",
                "./trainer/*",
                "./dataset/*",
                "./models/*"
            ]
        )
        neptune_callback["parameters"] = CONFIG


    ## Load dataset
    dataset_manager = DatasetManager(CONFIG)
    
    d_model_1 = Discriminator()
    d_model_2 = Discriminator()

    g_model_1 = Generator()
    g_model_2 = Generator()

    trainer = Cycle_Trainer(d_model_1, d_model_2, g_model_1, g_model_2, dataset_manager,
                      CONFIG, args.enable_log, neptune_callback)
    
    trainer.train_loop()


