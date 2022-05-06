import tensorflow as tf

''' CycleGAN Loss Functions '''
@tf.function
def generator_loss(x):
    mse = tf.keras.losses.MeanSquaredError()
    # bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    return mse(tf.ones_like(x), x)


@tf.function
def discriminator_loss(real, generated):
    mse = tf.keras.losses.MeanSquaredError()
    real_loss = mse(tf.ones_like(real), real)
    generated_loss = mse(tf.zeros_like(generated), generated)

    return (real_loss + generated_loss) * 0.5

@tf.function
def cycle_loss(real_x, cycled_x, c_lambda):
    loss = tf.reduce_mean(tf.abs(real_x - cycled_x))

    return c_lambda * loss

@tf.function
def identity_loss(real_x, same_x, i_lambda):
    loss = tf.reduce_mean(tf.abs(real_x - same_x))

    return i_lambda * loss