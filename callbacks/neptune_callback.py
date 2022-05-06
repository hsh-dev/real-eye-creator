from tensorflow.keras.callbacks import Callback


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
        
        # if logs.get("valid_loss") is not None:
        #     self.neptune["valid/generator_loss"].log(
        #         logs["valid_loss"])
                
