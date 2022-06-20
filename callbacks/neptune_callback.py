from tensorflow.keras.callbacks import Callback
from neptune.new.types import File

class NeptuneCallback(Callback):
    def __init__(self, neptune_instance):
        super(NeptuneCallback, self).__init__()
        self.neptune = neptune_instance

    def on_epoch_end(self, epoch, logs=None, lr=None):
        if logs.get("regularization_loss") is not None:
            self.neptune["train/reg_loss"].log(
                logs["regularization_loss"])
            
        if logs.get("generator_loss") is not None:
            self.neptune["train/gen_loss"].log(
                logs["generator_loss"])

        if logs.get("discriminator_real_loss") is not None:
            self.neptune["train/dis_real_loss"].log(
                logs["discriminator_real_loss"])

        if logs.get("discriminator_fake_loss") is not None:
            self.neptune["train/dis_fake_loss"].log(
                logs["discriminator_fake_loss"])

        if logs.get("learning_rate") is not None:
            self.neptune["train/learning_rate"].log(
                logs["learning_rate"]
            )
        
class Cycle_NeptuneCallback(Callback):
    def __init__(self, neptune_instance):
        super(Cycle_NeptuneCallback, self).__init__()
        self.neptune = neptune_instance

    def on_epoch_end(self, epoch, logs=None, lr=None):
        if logs.get("generator_g_loss") is not None:
            self.neptune["train/gen_g_loss"].log(
                logs["generator_g_loss"])
            
        if logs.get("generator_f_loss") is not None:
            self.neptune["train/gen_f_loss"].log(
                logs["generator_f_loss"])

        if logs.get("discriminator_x_loss") is not None:
            self.neptune["train/dis_x_loss"].log(
                logs["discriminator_x_loss"])

        if logs.get("discriminator_y_loss") is not None:
            self.neptune["train/dis_y_loss"].log(
                logs["discriminator_y_loss"])

        if logs.get("learning_rate") is not None:
            self.neptune["train/learning_rate"].log(
                logs["learning_rate"])

    def on_batch_end(self, step, logs):
        if logs.get["fake_image"] is not None:
            self.neptune["image/fake_image"].upload(logs["fake_image"])


        if logs.get["real_image"] is not None:
            self.neptune["image/real_image"].upload(logs["real_image"])
    

        if logs.get["generated_image"] is not None:
            self.neptune["image/generated_image"].upload(logs["generated_image"])
    
    
class Star_NeptuneCallback(Callback):
    def __init__(self, neptune_instance):
        super(Star_NeptuneCallback, self).__init__()
        self.neptune = neptune_instance

    def on_epoch_end(self, epoch, logs=None, lr=None):
        if logs.get("generator_loss") is not None:
            self.neptune["train/gen_loss"].log(
                logs["generator_loss"])

        if logs.get("discriminator_loss") is not None:
            self.neptune["train/dis_loss"].log(
                logs["discriminator_loss"])

        if logs.get("learning_rate") is not None:
            self.neptune["train/learning_rate"].log(
                logs["learning_rate"])
            
        if logs.get("total_train_accuracy") is not None:
            self.neptune["train/accuracy"].log(logs["total_train_accuracy"])        
        if logs.get("total_train_precision") is not None:
            self.neptune["train/precision"].log(logs["total_train_precision"])
        if logs.get("total_train_recall") is not None:
            self.neptune["train/recall"].log(logs["total_train_recall"])
        if logs.get("total_train_f1_score") is not None:
            self.neptune["train/f1_score"].log(logs["total_train_f1_score"])       



    def on_batch_end(self, step, logs):
        if logs.get["fake_image"] is not None:
            self.neptune["image/fake_image"].upload(logs["fake_image"])


        if logs.get["real_image"] is not None:
            self.neptune["image/real_image"].upload(logs["real_image"])
    

        if logs.get["generated_image"] is not None:
            self.neptune["image/generated_image"].upload(logs["generated_image"])
    