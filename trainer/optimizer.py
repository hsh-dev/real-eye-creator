import tensorflow as tf
import tensorflow.keras.backend as K
import math

class OptimizerWrapper:
    def __init__(self, optimizer, initial_lr, start_epoch, warmup_epochs=0, scheduler=None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.scheduler = scheduler
        self.start_epoch = start_epoch
        self.warmup_epochs = warmup_epochs
        
        assert self.optimizer.lr == self.initial_lr, "OptimizerWrapper: Optimizer should start with initial_lr"
        if self.scheduler is not None:
            assert self.scheduler(0) == self.initial_lr, "OptimizerWrapper: Scheduler should start with initial_lr"

    def get_lr(self, epoch):
        if epoch < self.start_epoch:
            return 0
        elif (self.warmup_epochs > 0) and (epoch - self.start_epoch < self.warmup_epochs):
            return self.initial_lr * (epoch - self.start_epoch + 1) / self.warmup_epochs
        else:
            if self.scheduler is None:
                return self.initial_lr
            return self.scheduler(epoch - self.start_epoch - self.warmup_epochs)

    def update_lr(self, epoch):
        new_lr = self.get_lr(epoch)
        K.set_value(self.optimizer.lr, new_lr)
        return new_lr

    def enable_mixed_precision(self):
        self.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self.optimizer)

class CosineDecayWrapper:
    # constructor
    def __init__(self, optimizer, max_lr, min_lr, n_epochs, n_cycles, decay_steps=None):
        self.optimizer = optimizer
        self.epochs = n_epochs
        self.cycles = n_cycles
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.decay_steps = decay_steps

    # caculate learning rate for an epoch
    def cosine_annealing(self, epoch):
        epochs_per_cycle = math.floor(self.epochs/self.cycles)
        cos_inner = (math.pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
        return self.max_lr/2 * (math.cos(cos_inner) + 1)

    # calculate and set learning rate at the start of the epoch
    def get_lr(self, epoch):
        if(epoch < self.decay_steps):
            lr = self.cosine_annealing(epoch)
        else:
            lr = self.min_lr
        return lr

    def update_lr(self, epoch):
        new_lr = self.get_lr(epoch)
        K.set_value(self.optimizer.lr, new_lr)
        return new_lr
