import tensorflow as tf
import numpy as np
from tqdm import tqdm
import sys
import time
import os
import cv2

from callbacks.neptune_callback import Cycle_NeptuneCallback
from utils.dir import check_make_dir
from utils.tools import tf_shuffle_axis
from trainer.optimizer import CosineDecayWrapper
from trainer.cyclegan_loss import generator_loss, discriminator_loss, cycle_loss, identity_loss

class Cycle_Trainer(object):
    def __init__(self, d_model_1, d_model_2, g_model_1, g_model_2, dataset_manager, config, enable_log = False, call_back = None):
        self.config = config
        
        self.generator_g = g_model_1
        self.generator_f = g_model_2
        self.discriminator_x = d_model_1
        self.discriminator_y = d_model_2

        self.c_lambda = 10
        self.i_lambda = 10

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
        self.neptune_callback = Cycle_NeptuneCallback(call_back)
        self.logs = {}
        
    def _init_optimizer(self):
        generator_g_optimizer = CosineDecayWrapper(
            tf.keras.optimizers.Adam(
                lr = self.initial_learning_rate, beta_1 = 0.5, beta_2 = 0.999),
            self.config
        )
        
        generator_f_optimizer = CosineDecayWrapper(
            tf.keras.optimizers.Adam(
                lr=self.initial_learning_rate, beta_1=0.5, beta_2=0.999),
            self.config
        )
        
        discriminator_x_optimizer = CosineDecayWrapper(
            tf.keras.optimizers.Adam(
                lr=self.initial_learning_rate, beta_1=0.99, beta_2=0.999),
            self.config
        )

        discriminator_y_optimizer = CosineDecayWrapper(
            tf.keras.optimizers.Adam(
                lr=self.initial_learning_rate, beta_1=0.99, beta_2=0.999),
            self.config
        )

        self.gen_g_optimizer = generator_g_optimizer
        self.gen_f_optimizer = generator_f_optimizer
        self.dis_x_optimizer = discriminator_x_optimizer
        self.dis_y_optimizer = discriminator_y_optimizer


    ''' Train Functions '''
    def train_loop(self):
        best_ = 1000
        epochs = range(1, int(self.epochs) + 1)      
        
        for epoch in epochs:
            print(f'[*] Start epoch {epoch}')
            self.epoch = epoch

            print(f'[*] Start train_model {epoch}')
            self.train_layer('train')

            print(f"Epoch: {self.epoch}\n")
            print(f'generator G loss : {self.logs["generator_g_loss"]}')
            print(f'generator F loss : {self.logs["generator_f_loss"]}')
            print(f'discriminator X loss : {self.logs["discriminator_x_loss"]}')
            print(f'discriminator Y loss : {self.logs["discriminator_y_loss"]}')

            self.lr = self.gen_g_optimizer.update_lr(self.epoch)
            self.lr = self.gen_f_optimizer.update_lr(self.epoch)
            self.lr = self.dis_x_optimizer.update_lr(self.epoch)
            self.lr = self.dis_y_optimizer.update_lr(self.epoch)

            if self.log_enabled:            
                self.neptune_callback.on_epoch_end(self.epoch, self.logs)
                
            self._save_weights(epoch)
            self._save_model(epoch)
            
            cur_gen_loss = (self.logs["generator_g_loss"] + self.logs["generator_f_loss"])*0.5
            if  best_ > cur_gen_loss :
                best_ = cur_gen_loss
                self._save_weights(epoch, best_)
                self._save_model(epoch, best_)


    def train_layer(self, phase):
        gen_g_losses = []
        gen_f_losses = []
        dis_x_losses = []
        dis_y_losses = []

        gen_g_loss_tmp = []
        gen_f_loss_tmp = []
        dis_x_loss_tmp = []
        dis_y_loss_tmp = []

        prev_time = time.time()

        dataset = self.dataset_manager.get_training_data(self.batch_size)
        steps = len(dataset)

        for step, sample in enumerate(dataset):
            self.total_step += 1

            real_x, real_y = sample

            with tf.GradientTape(persistent = True) as tape:
                # Generator G translates real_x -> fake_y
                # Generator F translates real_y -> fake_x
                # cycled_x : reconstructed x : F(G(x))
                # cycled_y : reconstructed y : G(F(y))
                # same_x, same_y : for identity loss

                fake_y = self.generator_g(real_x, training = True)
                cycled_x = self.generator_f(fake_y, training = True)

                fake_x = self.generator_f(real_y, training = True)
                cycled_y = self.generator_g(fake_x, training = True)
                same_x = self.generator_f(real_x, training = True)
                same_y = self.generator_g(real_y, training = True)

                dis_real_x = self.discriminator_x(real_x, training = True)
                dis_real_y = self.discriminator_y(real_y, training = True)

                dis_fake_x = self.discriminator_x(fake_x, training = True)
                dis_fake_y = self.discriminator_y(fake_y, training = True)

                # calculate loss
                gen_g_loss = generator_loss(dis_fake_y)
                gen_f_loss = generator_loss(dis_fake_x)

                total_cycle_loss = cycle_loss(real_x, cycled_x, self.c_lambda) + \
                    cycle_loss(real_y, cycled_y, self.c_lambda)

                # Generator Loss
                total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y, self.i_lambda)
                total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x, self.i_lambda)
                
                # Discriminator Loss
                dis_x_loss = discriminator_loss(dis_real_x, dis_fake_x)
                dis_y_loss = discriminator_loss(dis_real_y, dis_fake_y)
            
            
            # Calculate Gradients and Apply
            self._gradients(tape, total_gen_g_loss, total_gen_f_loss, dis_x_loss, dis_y_loss, step)
            del tape
            

            gen_g_losses.append(total_gen_g_loss)
            gen_g_loss_tmp.append(total_gen_g_loss)

            gen_f_losses.append(total_gen_f_loss)
            gen_f_loss_tmp.append(total_gen_f_loss)

            dis_x_losses.append(dis_x_loss)
            dis_x_loss_tmp.append(dis_x_loss)

            dis_y_losses.append(dis_y_loss)
            dis_y_loss_tmp.append(dis_y_loss)

            if (step+1) % 100 == 0:
                self.print_step(step+1, steps, 
                                gen_g_loss_tmp, 
                                gen_f_loss_tmp, 
                                dis_x_loss_tmp, 
                                dis_y_loss_tmp , prev_time)

                sys.stdout.flush()
                prev_time = time.time()
                gen_g_loss_tmp.clear()
                gen_f_loss_tmp.clear()
                dis_x_loss_tmp.clear()
                dis_y_loss_tmp.clear()
            
            if (step+1) % 200 == 0:
                self.save_image(real_x, real_y, self.epoch, step+1)

        
        self.logs["learning_rate"] = self.lr
        self.logs["generator_g_loss"] = sum(gen_g_losses) / len(gen_g_losses)
        self.logs["generator_f_loss"] = sum(gen_f_losses) / len(gen_f_losses)
        self.logs["discriminator_x_loss"] = sum(dis_x_losses) / len(dis_x_losses)
        self.logs["discriminator_y_loss"] = sum(dis_y_losses) / len(dis_y_losses)


    def _gradients(self, tape, total_gen_g_loss, total_gen_f_loss, dis_x_loss, dis_y_loss, step):
        generator_g_gradients = tape.gradient(total_gen_g_loss, self.generator_g.trainable_variables)
        generator_f_gradients = tape.gradient(total_gen_f_loss, self.generator_f.trainable_variables)
        discriminator_x_gradients = tape.gradient(dis_x_loss, self.discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(dis_y_loss, self.discriminator_y.trainable_variables)
        
        if (step % 40) < 20:
            self._apply_gradients(generator_g_gradients, generator_f_gradients)
        else:
            self._apply_gradients(generator_g_gradients, generator_f_gradients, discriminator_x_gradients, discriminator_y_gradients)

    def _apply_gradients(self, generator_g_gradients, generator_f_gradients, discriminator_x_gradients = None, discriminator_y_gradients = None):
        self.gen_g_optimizer.optimizer.apply_gradients(
            zip(generator_g_gradients, self.generator_g.trainable_variables))

        self.gen_f_optimizer.optimizer.apply_gradients(
            zip(generator_f_gradients, self.generator_f.trainable_variables))

        if discriminator_x_gradients is not None:
            self.dis_x_optimizer.optimizer.apply_gradients(
                zip(discriminator_x_gradients, self.discriminator_x.trainable_variables))

        if discriminator_y_gradients is not None:
            self.dis_y_optimizer.optimizer.apply_gradients(
                zip(discriminator_y_gradients, self.discriminator_y.trainable_variables))


    ''' Image Show '''
    def save_image(self, real_x, real_y, epoch, step):
        # dataset = self.dataset_manager.get_one_training_data()
        tag = "ep_" + str(epoch) + "_step_" + str(step)
        save_image_path = os.path.join(self.save_path, "image")
        check_make_dir(save_image_path)

        generated = self.generator_g(real_x)

        # for step, sample in enumerate(dataset):
        real_x = np.squeeze(real_x.numpy(), axis = 0)*255
        fake_image_name = os.path.join(save_image_path, tag + "_fake_sample.jpg")
        cv2.imwrite(fake_image_name, real_x)

        real_y = np.squeeze(real_y.numpy(), axis=0)*255
        real_image_name = os.path.join(save_image_path, tag + "_real_sample.jpg")
        cv2.imwrite(real_image_name, real_y)

        generated = np.squeeze(generated.numpy(), axis = 0)*255
        generated_image_name = os.path.join(save_image_path, tag + "_generated_sample.jpg")
        cv2.imwrite(generated_image_name, generated)

        self.logs["fake_image"] = fake_image_name
        self.logs["real_image"] = real_image_name
        self.logs["generated_image"] = generated_image_name




        

    ''' Save Functions '''
    def _save_model(self, epoch, best_ = None):
        if best_ == None:
            save_model_path=os.path.join(self.save_path, "saved_model")
            check_make_dir(save_model_path)

            model_name = f'generator_g_{epoch}'
            model_path = os.path.join(save_model_path, model_name)
            self.generator_g.save(model_path, overwrite=True, save_format="tf")

            model_name = f'generator_f_{epoch}'
            model_path = os.path.join(save_model_path, model_name)
            self.generator_f.save(model_path, overwrite=True, save_format="tf")

            model_name = f'discriminator_x_{epoch}'
            model_path = os.path.join(save_model_path, model_name)
            self.discriminator_x.save(model_path, overwrite=True, save_format="tf")

            model_name = f'discriminator_y_{epoch}'
            model_path = os.path.join(save_model_path, model_name)
            self.discriminator_y.save(model_path, overwrite=True, save_format="tf")

            print(f'Saved model of epoch {epoch}')

        else :
            save_model_path=os.path.join(self.save_path, "saved_model")
            check_make_dir(save_model_path)

            model_name = f'generator_g_best'
            model_path = os.path.join(save_model_path, model_name)
            self.generator_g.save(model_path, overwrite=True, save_format="tf")

            model_name = f'generator_f_best'
            model_path = os.path.join(save_model_path, model_name)
            self.generator_f.save(model_path, overwrite=True, save_format="tf")

            model_name = f'discriminator_x_best'
            model_path = os.path.join(save_model_path, model_name)
            self.discriminator_x.save(
                model_path, overwrite=True, save_format="tf")

            model_name = f'discriminator_y_best'
            model_path = os.path.join(save_model_path, model_name)
            self.discriminator_y.save(
                model_path, overwrite=True, save_format="tf")

            print(f'Saved model of epoch {epoch}')
            
    def _save_weights(self, epoch, best_ = None):
        if best_ == None:
            save_weights_path=os.path.join(self.save_path, "checkpoints")
            check_make_dir(save_weights_path)

            model_name = f'generator_g_{epoch}.tf'
            model_path = os.path.join(save_weights_path, model_name)
            self.generator_g.save_weights(model_path)

            model_name = f'generator_f_{epoch}.tf'
            model_path = os.path.join(save_weights_path, model_name)
            self.generator_f.save_weights(model_path)

            model_name = f'discriminator_x_{epoch}.tf'
            model_path = os.path.join(save_weights_path, model_name)
            self.discriminator_x.save_weights(model_path)

            model_name = f'discriminator_y_{epoch}.tf'
            model_path = os.path.join(save_weights_path, model_name)
            self.discriminator_y.save_weights(model_path)

            print(f'Saved model weights of epoch {epoch}')

        else :
            save_weights_path=os.path.join(self.save_path, "checkpoints")
            check_make_dir(save_weights_path)

            model_name = f'generator_g_best.tf'
            model_path = os.path.join(save_weights_path, model_name)
            self.generator_g.save_weights(model_path)

            model_name = f'generator_f_best.tf'
            model_path = os.path.join(save_weights_path, model_name)
            self.generator_f.save_weights(model_path)

            model_name = f'discriminator_x_best.tf'
            model_path = os.path.join(save_weights_path, model_name)
            self.discriminator_x.save_weights(model_path)

            model_name = f'discriminator_y_best.tf'
            model_path = os.path.join(save_weights_path, model_name)
            self.discriminator_y.save_weights(model_path)
            
            print(f'Saved best model weights of epoch {epoch}')
            

    def free(self):
        self.validation_set = None
        del self.validation_set
        
    def print_step(self, curr_step, total_steps, 
                    Gen_G_Loss, 
                    Gen_F_Loss, 
                    Dis_X_Loss, 
                    Dis_Y_Loss, prev_time):
        gen_g_out = sum(Gen_G_Loss)/len(Gen_G_Loss)
        gen_f_out = sum(Gen_F_Loss)/len(Gen_F_Loss)
        dis_x_out = sum(Dis_X_Loss)/len(Dis_X_Loss)
        dis_y_out = sum(Dis_Y_Loss)/len(Dis_Y_Loss)

        print("[TRAIN] Epoch: {} | Step: {}/{} | Time elapsed: {} "\
            .format(self.epoch, curr_step, total_steps, time.time()-prev_time))
        print("|Gen G Loss : {} |Gen F Loss: {} |Dis X Loss: {} |Dis Y Loss: {} | "\
            .format(gen_g_out, gen_f_out, dis_x_out, dis_y_out))
    
    def print_step_valid(self, curr_step, total_steps, G_Loss, prev_time):
        print("[VALID] Epoch: {} | Step: {}/{} | Dis.Loss: {} | Gen.Loss: {} | Time elapsed: {}".format(self.epoch, curr_step,
              total_steps, sum(G_Loss)/len(G_Loss), time.time()-prev_time))


