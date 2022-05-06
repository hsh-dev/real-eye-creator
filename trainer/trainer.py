import tensorflow as tf
import numpy as np
from tqdm import tqdm
import sys
import time
import os

from callbacks.neptune_callback import NeptuneCallback
from utils.dir import check_make_dir
from utils.tools import tf_shuffle_axis
from trainer.optimizer import CosineDecayWrapper

class Trainer(object):
    def __init__(self, d_model, g_model, dataset_manager, config, enable_log = False, call_back = None):
        self.discriminator = d_model
        self.generator = g_model

        self.save_path = config['save_path']
        self.batch_size = config['batch_size']
        self.val_batch_size = config['val_batch_size']
        self.log_enabled = enable_log
        self.epochs = config['epochs']
        self.cycle = config['cycle']
        self.decay_steps = config['decay_steps']

        self.initial_learning_rate = config['learning_rate']
        self.min_learning_rate = config['min_learning_rate']
        self.lr = self.initial_learning_rate
        
        self.dataset_manager = dataset_manager
        
        self.total_step = 0
        self.logs = {}
        
        self.decay_start_epochs = 10
        
        self._init_optimizer()

        if enable_log:
            self._init_callbacks(call_back)


    ''' Initialize Functions '''   
    def _init_callbacks(self, call_back):
        self.neptune_callback = NeptuneCallback(call_back)
        self.logs = {}
        
    def _init_optimizer(self):
        g_optimizer = CosineDecayWrapper(
            tf.keras.optimizers.Adam(
                lr = self.initial_learning_rate, beta_1 = 0.99, beta_2 = 0.999),
            self.initial_learning_rate,
            self.min_learning_rate,
            self.epochs,
            self.cycle,
            self.decay_steps
        )
        
        d_optimizer = CosineDecayWrapper(
            tf.keras.optimizers.Adam(
                lr=self.initial_learning_rate, beta_1=0.99, beta_2=0.999),
            self.initial_learning_rate,
            self.min_learning_rate,
            self.epochs,
            self.cycle,
            self.decay_steps
        )
        
        sgd_optimizer = CosineDecayWrapper(
            tf.keras.optimizers.SGD(self.initial_learning_rate, 0.99),
            self.initial_learning_rate,
            self.min_learning_rate,
            self.epochs,
            self.cycle*2,
            self.decay_steps
        )

        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.r_optimizer = sgd_optimizer


    ''' Loss Functions '''
    def regularization_loss(self, fake, fake_generated):
        ''' 
        Compute L1 loss between fake image and fake generated image \n
        gamma : scale constant
        '''
        gamma = 10
        L1_loss = tf.reduce_mean(tf.abs(fake - fake_generated))
        reg_loss = L1_loss * gamma

        return reg_loss
    
    def local_advertial_loss(self, y_pred, type):
        y_true = None
        if type == 'real':
            y_true = tf.ones([y_pred.shape[0], y_pred.shape[1]])
        else:
            y_true = tf.zeros([y_pred.shape[0], y_pred.shape[1]])

        advertial_loss = self.custom_bce_loss(y_pred, y_true)

        advertial_loss = advertial_loss / y_pred.shape[0]

        return advertial_loss

    @tf.function
    def mean_squared_loss(self, y_pred, type):
        '''
        Compute mean squared loss
        '''
        mse = tf.keras.losses.MeanSquaredError()
        y_true = None
        if type == 'real':
            y_true = tf.ones([y_pred.shape[0], y_pred.shape[1]])
        else:
            y_true = tf.zeros([y_pred.shape[0], y_pred.shape[1]])
            
        return mse(y_true, y_pred)

    def discriminative_loss(self, y_real, y_fake):
        loss = tf.nn.softmax_cross_entropy_with_logits(y_real, y_fake)
        return loss


    @tf.function
    def custom_bce_loss(self, y_true, y_pred):
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        bce_loss= bce(y_true, y_pred)
        return bce_loss

      
    ''' Train Functions '''
    def train_loop(self):
        best_ = 1000
        epochs = range(1, int(self.epochs) + 1)      
        
        for epoch in epochs:
            print(f'[*] Start epoch {epoch}')
            self.epoch = epoch

            print(f'[*] Start train_model {epoch}')
            self.train_layer('train')
        
            # print(f'[*] Start validate_model {epoch}')
            # cur_loss = self.valid_loop()

            print(f"Epoch: {self.epoch}\n")
            print(f'regularization loss : {self.logs["regularization_loss"]}')
            print(f'generator loss : {self.logs["generator_loss"]}')
            print(f'discriminator real loss : {self.logs["discriminator_real_loss"]}')
            print(f'discriminator fake loss : {self.logs["discriminator_fake_loss"]}')
            # print(f'validation loss : {self.logs["valid_loss"]}')

            self.lr = self.r_optimizer.update_lr(self.epoch)
            self.lr = self.g_optimizer.update_lr(self.epoch)
            self.lr = self.d_optimizer.update_lr(self.epoch)
            
            if self.log_enabled:            
                self.neptune_callback.on_epoch_end(self.epoch, self.logs)
                
            self._save_weights(epoch)
            self._save_model(epoch)
            
            # if  best_ > cur_loss :
            #     best_ = cur_loss
            #     self._save_weights(epoch, best_)
            #     self._save_model(epoch, best_)

    def train_layer(self, phase):
        reg_losses = []
        gen_losses = []
        dis_real_losses = []
        dis_fake_losses = []

        reg_loss_tmp = []
        dis_real_tmp = []
        dis_fake_tmp = []
        gen_loss_tmp = []

        prev_time = time.time()

        dataset = self.dataset_manager.get_training_data(self.batch_size)
        steps = len(dataset)

        buffer_dataset = None

        for step, sample in enumerate(dataset):
            self.total_step += 1
            fake_base, real_image = sample

            # ===== Refine Generator first ===== #
            tmp = []
            for i in range(10):
                with tf.GradientTape(persistent=True) as tape:
                    fake_generated = self.generator(fake_base, training=True)
                    reg_loss = self.regularization_loss(fake_base, fake_generated)
                    r_gradients = tape.gradient(reg_loss, self.generator.trainable_variables)
                    self.r_optimizer.optimizer.apply_gradients(zip(r_gradients, self.generator.trainable_variables))
                del tape
                
                tmp.append(reg_loss)
                tf.debugging.check_numerics(reg_loss, 'regularization loss is nan')
            reg_loss_ = np.mean(tmp)


            with tf.GradientTape(persistent=True) as generator_tape:
                # ====== Train Generator ===== #
                fake_generated = self.generator(fake_base, training=True)
                fake_output = self.discriminator(fake_generated, training=True)
                g_loss = self.mean_squared_loss(fake_output, 'real')
                g_gradients = generator_tape.gradient(g_loss, self.generator.trainable_variables)
                self.g_optimizer.optimizer.apply_gradients(
                    zip(g_gradients, self.generator.trainable_variables))
            del generator_tape

            # use buffer dataset
            # N/2 fake_output go into buffer dataset (N : batch size)
            if step == 0:
                buffer_dataset = fake_generated
            elif fake_generated.shape[0] != 1:
                N = int(self.batch_size/2)

                temp_dataset_1, temp_dataset_2 = tf.split(fake_generated, 2, 0)
                buffer_1, buffer_2 = tf.split(buffer_dataset, 2, 0)

                fake_generated = tf.concat([temp_dataset_1, buffer_2], 0)
                buffer_dataset = tf.concat([temp_dataset_2, buffer_1], 0)
                
                fake_generated = tf_shuffle_axis(fake_generated)
                buffer_dataset = tf_shuffle_axis(buffer_dataset)


            with tf.GradientTape(persistent=True) as discriminator_tape:
                # ======= Train Discriminator ======= #
                real_output = self.discriminator(real_image, training = True)
                d_loss_real = self.mean_squared_loss(real_output, 'real')
                
                fake_output = self.discriminator(fake_generated, training=True)
                d_loss_fake = self.mean_squared_loss(fake_output, 'fake')

                d_loss = (d_loss_real + d_loss_fake)*0.5
                
                d_gradients = discriminator_tape.gradient(d_loss, self.discriminator.trainable_variables)
                self.d_optimizer.optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))

            del discriminator_tape
            
            reg_losses.append(reg_loss_)
            reg_loss_tmp.append(reg_loss_)

            dis_real_losses.append(d_loss_real)
            tf.debugging.check_numerics(d_loss_real, 'discriminator real loss is nan')
            dis_real_tmp.append(d_loss_real)

            dis_fake_losses.append(d_loss_fake)
            tf.debugging.check_numerics(d_loss_fake, 'discriminator fake loss is nan')
            dis_fake_tmp.append(d_loss_fake)

            gen_losses.append(g_loss)
            tf.debugging.check_numerics(g_loss, 'generator loss is nan')
            gen_loss_tmp.append(g_loss)

            if (step+1) % 10 == 0:
                self.print_step(step+1, steps, reg_loss_tmp, dis_real_tmp, dis_fake_tmp, gen_loss_tmp, prev_time)
                sys.stdout.flush()
                prev_time = time.time()
                reg_loss_tmp.clear()
                dis_real_tmp.clear()
                dis_fake_tmp.clear()
                gen_loss_tmp.clear()
        
        self.logs["learning_rate"] = self.lr
        self.logs["regularization_loss"] = sum(reg_losses) / len(reg_losses)
        self.logs["generator_loss"] = sum(gen_losses)/len(gen_losses)
        self.logs["discriminator_real_loss"] = sum(dis_real_losses)/len(dis_real_losses)
        self.logs["discriminator_fake_loss"] = sum(dis_fake_losses)/len(dis_fake_losses)



    ''' Save Functions '''
    def _save_model(self, epoch, best_ = None):
        if best_ == None:
            save_model_path=os.path.join(self.save_path, "saved_model")
            check_make_dir(save_model_path)

            model_name = f'generator_{epoch}'
            model_path = os.path.join(save_model_path, model_name)
            self.generator.save(model_path, overwrite=True, save_format="tf")

            model_name = f'discriminator_{epoch}'
            model_path = os.path.join(save_model_path, model_name)
            self.generator.save(model_path, overwrite=True, save_format="tf")
            print(f'Saved model of epoch {epoch}')

        else :
            save_model_path=os.path.join(self.save_path, "saved_model")
            check_make_dir(save_model_path)

            model_name = f'generator_best'
            model_path = os.path.join(save_model_path, model_name)
            self.generator.save(model_path, overwrite=True, save_format="tf")

            model_name = f'discriminator_best'
            model_path = os.path.join(save_model_path, model_name)
            self.discriminator.save(model_path, overwrite=True, save_format="tf")
            print(f'Saved model of epoch {epoch}')
            
    def _save_weights(self, epoch, best_ = None):
        if best_ == None:
            save_weights_path=os.path.join(self.save_path, "checkpoints")
            check_make_dir(save_weights_path)

            model_name = f'generator_{epoch}.tf'
            model_path = os.path.join(save_weights_path, model_name)
            self.generator.save_weights(model_path)

            model_name = f'discriminator_{epoch}.tf'
            model_path = os.path.join(save_weights_path, model_name)
            self.discriminator.save_weights(model_path)

            print(f'Saved model weights of epoch {epoch}')

        else :
            save_weights_path=os.path.join(self.save_path, "checkpoints")
            check_make_dir(save_weights_path)

            model_name = f'generator.tf'
            model_path = os.path.join(save_weights_path, model_name)
            self.generator.save_weights(model_path)
            
            model_name = f'discriminator_best.tf'
            model_path = os.path.join(save_weights_path, model_name)
            self.discriminator.save_weights(model_path)
            
            print(f'Saved best model weights of epoch {epoch}')
            

    def free(self):
        self.validation_set = None
        del self.validation_set
        
    def print_step(self, curr_step, total_steps, Reg_Loss, D_R_Loss, D_F_Loss, G_Loss, prev_time):
        reg_out = sum(Reg_Loss)/len(Reg_Loss)
        dis_r_out = sum(D_R_Loss)/len(D_R_Loss)
        dis_f_out = sum(D_F_Loss)/len(D_F_Loss)
        gen_out = sum(G_Loss)/len(G_Loss)

        print("[TRAIN] Epoch: {} | Step: {}/{} | Time elapsed: {} "\
            .format(self.epoch, curr_step, total_steps, time.time()-prev_time))
        print("| Reg Loss : {} | Dis Real Loss: {} | Dis Fake Loss: {} | Gen Loss: {} | "\
            .format(reg_out, dis_r_out, dis_f_out, gen_out))
    
    def print_step_valid(self, curr_step, total_steps, G_Loss, prev_time):
        print("[VALID] Epoch: {} | Step: {}/{} | Dis.Loss: {} | Gen.Loss: {} | Time elapsed: {}".format(self.epoch, curr_step,
              total_steps, sum(G_Loss)/len(G_Loss), time.time()-prev_time))


