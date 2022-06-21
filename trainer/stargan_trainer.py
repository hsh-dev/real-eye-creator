from numpy import gradient
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import sys
import time
import os
import cv2
from random import *
from sklearn.metrics import confusion_matrix

from utils.dir import check_make_dir
from trainer.optimizer import CosineDecayWrapper

from callbacks.neptune_callback import Star_NeptuneCallback
from trainer.stargan_loss import classification_loss, state_loss, identity_loss, adversarial_loss

class Star_Trainer(object):
    def __init__(self, d_model, g_model, dataset_manager, config, enable_log = False, call_back = None):
        self.config = config
        
        self.generator = g_model
        self.discriminator = d_model

        self.cls_lambda = 10
        self.rec_lambda = 5
        self.gp_lambda = 5
        self.sta_lambda = 10
        
        self.domain_number = 3
        
        self.save_path = config['save_path']
        
        self.batch_size = config['batch_size']
        self.val_batch_size = config['val_batch_size']
        self.test_batch_size = config['test_batch_size']

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
        self.neptune_callback = Star_NeptuneCallback(call_back)
        self.logs = {}
        
    def _init_optimizer(self):
        generator_optimizer = CosineDecayWrapper(
            optimizer  = tf.keras.optimizers.Adam(
                learning_rate = self.initial_learning_rate, beta_1 = 0.5, beta_2 = 0.999),
            max_lr = self.initial_learning_rate,
            min_lr = self.min_learning_rate,
            max_epochs = self.epochs,
            decay_cycles = self.cycle,
            decay_epochs = self.decay_steps
        )
        
        discriminator_optimizer = CosineDecayWrapper(
            tf.keras.optimizers.Adam(
                learning_rate=self.initial_learning_rate, beta_1=0.99, beta_2=0.999),
            max_lr = self.initial_learning_rate,
            min_lr = self.min_learning_rate,
            max_epochs = self.epochs,
            decay_cycles = self.cycle,
            decay_epochs = self.decay_steps
        )

        self.gen_optimizer = generator_optimizer
        self.dis_optimizer = discriminator_optimizer


    ''' Train Functions '''
    def train_loop(self):
        best_ = 1000
        epochs = range(1, int(self.epochs) + 1)      
        
        for epoch in epochs:
            print(f'[*] Start epoch {epoch}')
            self.epoch = epoch

            print(f'[*] Start train_model {epoch}')
            self.train_layer('train')
            
            print("______________________")
            print(f"|*| Epoch: {self.epoch}")
            print(f'|*| Generator loss : {self.logs["generator_loss"]}')
            print(f'|*| Discriminator loss : {self.logs["discriminator_loss"]}')
            print("______________________")
            print(f'|*| Accuracy : {self.logs["total_train_accuracy"]}')
            print(f'|*| Precision : {self.logs["total_train_precision"]}')
            print(f'|*| Recall : {self.logs["total_train_recall"]}')
            print(f'|*| F1 Score : {self.logs["total_train_f1_score"]}')
            
            self.lr = self.gen_optimizer.update_lr(self.epoch)
            self.lr = self.dis_optimizer.update_lr(self.epoch)

            if self.log_enabled:            
                self.neptune_callback.on_epoch_end(self.epoch, self.logs)
                
            self._save_weights(epoch)
            self._save_model(epoch)
            
            cur_gen_loss = self.logs["generator_loss"]
            if  best_ > cur_gen_loss :
                best_ = cur_gen_loss
                self._save_weights(epoch, best_)
                self._save_model(epoch, best_)


    def train_layer(self, phase):
        gen_losses = []
        dis_losses = []

        gen_loss_tmp = []
        dis_loss_tmp = []

        prev_time = time.time()

        dataset = None
        if phase == "train":
            dataset = self.dataset_manager.get_training_data(self.batch_size)
        elif phase == "val":
            dataset = self.dataset_manager.get_validation_data(self.val_batch_size)
        elif phase == "test":
            dataset = self.dataset_manager.get_test_data(self.test_batch_size)        
        steps = len(dataset)

        total_cm = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}

        for step, sample in enumerate(dataset):
            self.total_step += 1

            real_img, real_s, real_d = sample
            real_s = tf.reshape(real_s, [real_s.shape[0], 1])
            
            with tf.GradientTape(persistent=True) as tape:                
                '''
                1. Discriminator
                '''
                dis_real_out, dis_real_d, dis_real_s = self.discriminator(real_img)
                real_cls_loss = classification_loss(real_d, dis_real_d)
                real_sta_loss = state_loss(real_s, dis_real_s)                
                
                '''
                2. Generator
                target domain -> random one hot vector
                target state -> same with orginal state
                '''
                batch = real_img.shape[0]
                target_d = random_onehotvector(batch, self.domain_number)
                target_c = tf.concat([target_d, real_s], axis=1)
                target_c = self.make_generator_input(real_img, target_c)

                fake_img = self.generator(real_img, target_c)
                dis_fake_out, dis_fake_d, dis_fake_s = self.discriminator(fake_img)
                
                fake_cls_loss = classification_loss(dis_fake_d, real_d)
                fake_sta_loss = state_loss(real_s, dis_fake_s)
                
                '''
                3. Generator Reconstruction
                Generated Image -> Origin Class (Domain + State)
                '''
                origin_c = tf.concat([real_d, real_s], axis=1)
                origin_c = self.make_generator_input(fake_img, origin_c)
                fake_rec_img = self.generator(fake_img, origin_c)
                
                rec_loss = identity_loss(fake_img, fake_rec_img, self.rec_lambda)
                
                '''
                4. Calculate Loss
                '''
                gp_loss = self.gradient_penalty(batch, real_img, fake_img)
                adv_loss = adversarial_loss(
                    dis_real_out,
                    dis_fake_out,
                    gp_loss,
                    self.gp_lambda)
                
                dis_loss = -adv_loss + self.cls_lambda * real_cls_loss + self.sta_lambda * real_sta_loss
                gen_loss = adv_loss + self.cls_lambda * fake_cls_loss + self.sta_lambda * fake_sta_loss + rec_loss
            
            # Calculate Gradients and Apply
            self._gradients(tape, gen_loss, dis_loss, step)
            del tape
            

            gen_losses.append(gen_loss)
            gen_loss_tmp.append(gen_loss)

            dis_losses.append(dis_loss)
            dis_loss_tmp.append(dis_loss)

            cm_list = self.custom_acc_(real_s, dis_real_s)
            for i, key in enumerate(total_cm):
                total_cm[key] += cm_list[i]

            if (step+1) % 20 == 0:
                self.print_step(step+1, steps, 
                                gen_loss_tmp, 
                                dis_loss_tmp , prev_time)
                self.save_image(real_img, fake_img, self.epoch, step)

                sys.stdout.flush()
                prev_time = time.time()
                gen_loss_tmp.clear()
                dis_loss_tmp.clear()
                
            # if (step+1) % 200 == 0:
            #     self.save_image(real_x, real_y, self.epoch, step+1)
                
        metrics = self.cal_acc(total_cm)
        (accuracy_, precision_, recall_, f1_s_) = metrics

        self.logs["learning_rate"] = self.lr
        self.logs["generator_loss"] = sum(gen_losses) / len(gen_losses)
        self.logs["discriminator_loss"] = sum(dis_losses) / len(dis_losses)
        
        self.logs["total_train_accuracy"] = accuracy_
        self.logs["total_train_precision"] = precision_
        self.logs["total_train_recall"] = recall_
        self.logs["total_train_f1_score"] = f1_s_


    ''' Gradient Functions ''' 
    def _gradients(self, tape, gen_loss, dis_loss, step):
        generator_gradients = tape.gradient(gen_loss, self.generator.trainable_variables)
        discriminator_gradients = tape.gradient(dis_loss, self.discriminator.trainable_variables)
        
        self._apply_gradients(generator_gradients, discriminator_gradients)


    def _apply_gradients(self, generator_gradients, discriminator_gradients):
        self.gen_optimizer.optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_variables))

        self.dis_optimizer.optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables))


    ''' Image Show '''
    def save_image(self, real_img, fake_img, epoch, step):

        one_real_img = real_img[0]
        one_fake_img = fake_img[0]
        # dataset = self.dataset_manager.get_one_training_data()
        tag = "ep_" + str(epoch) + "_step_" + str(step)
        save_image_path = os.path.join(self.save_path, "image")
        check_make_dir(save_image_path)

        one_real_img = one_real_img.numpy()*255
        one_real_img = cv2.cvtColor(one_real_img, cv2.COLOR_RGB2BGR)
        real_image_name = os.path.join(save_image_path, tag + "_real_sample.jpg")
        cv2.imwrite(real_image_name, one_real_img)
        

        one_fake_img = one_fake_img.numpy()*255
        one_fake_img = cv2.cvtColor(one_fake_img, cv2.COLOR_RGB2BGR)
        fake_image_name = os.path.join(save_image_path, tag + "_fake_sample.jpg")
        cv2.imwrite(fake_image_name, one_fake_img)

        self.logs["fake_image"] = fake_img
        self.logs["real_image"] = real_img

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
            self.discriminator.save(model_path, overwrite=True, save_format="tf")

            print(f'Saved model of epoch {epoch}')

        else :
            save_model_path=os.path.join(self.save_path, "saved_model")
            check_make_dir(save_model_path)

            model_name = f'generator_best'
            model_path = os.path.join(save_model_path, model_name)
            self.generator.save(model_path, overwrite=True, save_format="tf")

            model_name = f'discriminator_best'
            model_path = os.path.join(save_model_path, model_name)
            self.discriminator.save(
                model_path, overwrite=True, save_format="tf")

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

            model_name = f'generator_best.tf'
            model_path = os.path.join(save_weights_path, model_name)
            self.generator.save_weights(model_path)

            model_name = f'discriminator_best.tf'
            model_path = os.path.join(save_weights_path, model_name)
            self.discriminator.save_weights(model_path)
            
            print(f'Saved best model weights of epoch {epoch}')
            

    def free(self):
        self.validation_set = None
        del self.validation_set
    
    '''Print Functions'''
    def print_step(self, curr_step, total_steps, 
                    Gen_Loss, 
                    Dis_Loss, 
                    prev_time):
        gen_out = sum(Gen_Loss)/len(Gen_Loss)
        dis_out = sum(Dis_Loss)/len(Dis_Loss)

        print("[TRAIN] Epoch: {} | Step: {}/{} | Time elapsed: {} "\
            .format(self.epoch, curr_step, total_steps, time.time()-prev_time))
        print("|Gen Loss : {} |Dis Loss: {} | ".format(gen_out, dis_out))
    
    def print_step_valid(self, curr_step, total_steps, G_Loss, prev_time):
        print("[VALID] Epoch: {} | Step: {}/{} | Dis.Loss: {} | Gen.Loss: {} | Time elapsed: {}".format(self.epoch, curr_step,
              total_steps, sum(G_Loss)/len(G_Loss), time.time()-prev_time))


    '''Util Functions'''
    def make_generator_input(self, img, c):
        c_block = c
        c_block = tf.reshape(c_block, (c_block.shape[0], 1, 1, -1))
        c_block = tf.repeat(input = c_block, repeats=[img.shape[1]], axis=1)
        c_block = tf.repeat(input = c_block, repeats=[img.shape[2]], axis=2)
        return c_block
    
    def gradient_penalty(self, batch_size, real_images, fake_images):
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff
        
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred, a, b = self.discriminator(interpolated, training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        del gp_tape
        
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp
    
    def custom_acc_(self, labels, model_outputs):
        labels = tf.cast(labels, tf.float32)
        model_outputs = tf.cast(model_outputs, tf.float32)
        
        model_outputs = tf.where( tf.less_equal( 0.5, model_outputs ), 1, 0 )
        model_outputs = tf.reshape(model_outputs, labels.shape)
        
        labels = tf.where( tf.less_equal( 0.5, labels ), 1, 0 )
        
        cm = confusion_matrix(labels, model_outputs, labels=[0, 1])
        return [cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]]

    def cal_acc(self, total_cm):
        try:
            Precision = total_cm['tp'] / (total_cm['fp']+ total_cm['tp'])
        except ZeroDivisionError:
            Precision = 0            
        try:
            Recall = total_cm['tp'] / (total_cm['tp'] + total_cm['fn'])
        except ZeroDivisionError:
            Recall = 0            
        try:
            Accuracy = (total_cm['tp'] + total_cm['tn']) / (total_cm['tp'] + total_cm['fn'] + total_cm['fp'] + total_cm['tn'])
        except ZeroDivisionError:
            Accuracy = 0            
        try:
            F1_s = 2 *( (Precision * Recall) / (Precision + Recall) )
        except ZeroDivisionError:
            F1_s = 0        
        return Accuracy, Precision, Recall, F1_s



def random_onehotvector(batch, length):
    vector_set = np.eye(length, dtype = int)
    output = np.empty((0, length), int)
    
    for i in range(batch):
        rand_num = randint(0, length-1)
        output = np.append(output, np.array([vector_set[rand_num]]), axis=0)
    
    out_tensor = tf.convert_to_tensor(output, dtype=tf.float32)

    return out_tensor
