import tensorflow as tf
import argparse
import neptune.new as neptune
import os

from models.real_eye_gan import Discriminator, Generator
from models.star_gan_model import Star_Discriminator, Star_Generator

from dataset.dataset_manager import DatasetManager
from dataset.gan_dataset_manager import Gan_DatasetManager
from trainer.stargan_trainer import Star_Trainer


from utils.dir import check_make_dir
from utils.gpu import gpu_init
from utils.neptune_init import neptune_initialize
from utils.arg import arg_init
from config_file.config import CONFIG

if __name__ == "__main__":
    # GPU, Args, Neptune Init
    gpu_init()
    args, args_CONFIG = arg_init(CONFIG)
    neptune_callback = neptune_initialize(args, args_CONFIG)

    dataset_manager = Gan_DatasetManager(args_CONFIG)
    dataset_manager = DatasetManager(args_CONFIG)
    
    '''
    StarGan Implementation
    '''

    g_model = Star_Generator()
    d_model = Star_Discriminator()
    
    trainer = Star_Trainer(d_model, g_model, dataset_manager, args_CONFIG, args.enable_log, neptune_callback)
    
    trainer.train_loop()