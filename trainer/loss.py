import tensorflow as tf

''' Loss Functions '''
def regularization_loss(fake, fake_generated, gamma):
    ''' 
    Compute L1 loss between fake image and fake generated image \n
    gamma : scale constant
    '''
    L1_loss = tf.reduce_mean(tf.abs(fake - fake_generated))
    reg_loss = L1_loss * gamma

    return reg_loss

def local_advertial_loss(y_pred, type):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    y_true = None
    if type == 'real':
        y_true = tf.ones_like(y_pred)
    else:
        y_true = tf.zeros_like(y_pred)

    return bce(y_true, y_pred)


@tf.function
def mean_squared_loss(y_pred, type):
    '''
    Compute mean squared loss
    '''
    mse = tf.keras.losses.MeanSquaredError()
    y_true = None
    if type == 'real':
        y_true = tf.ones_like(y_pred)
    else:
        y_true = tf.zeros_like(y_pred)
        
    return mse(y_true, y_pred)

@tf.function
def discriminative_loss(y_real, y_fake):
    loss = tf.nn.softmax_cross_entropy_with_logits(y_real, y_fake)
    return loss
