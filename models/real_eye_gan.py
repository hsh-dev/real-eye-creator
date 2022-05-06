import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Flatten,MaxPool2D,\
    DepthwiseConv2D, Dropout, GlobalAveragePooling2D, ReLU, Conv2DTranspose, LeakyReLU, Softmax


class RE_Discriminator(Model):
    def __init__(self):
        super().__init__()
        # 36x60x3 -> 18x30x64
        self.conv1 = Conv2D(64, 4, 2, 'same')

        # 18x30x64 -> 9x15x128
        self.conv2 = Down(128, 4, 2, 'same')

        # 9x15x128 -> 4x7x256
        self.conv3 = Down(256, 3, 2, 'valid')

        # 4x7x256 -> 2x5x512
        self.conv4 = Down(512, 3, 1, 'valid')

        # 2x5x512 -> 2x5x1
        self.conv5 = Down(1, 1, 1, 'valid')
        
        # 10x1 -> 1x1
        self.outlayer = Sequential([
            Flatten(),
            Dense(1, activation = 'relu')
        ])
        
    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.outlayer(x)

        return x

class Down(Model):
    def __init__(self, n_out, kernel, stride, padding):
        super().__init__()
        self.conv = Sequential([
            LeakyReLU(0.1),
            Conv2D(n_out, kernel, stride, padding),
            BatchNormalization(epsilon=1e-3, momentum=0.999)
        ])

    def call(self,x):
        return self.conv(x)

class Up(Model):
    def __init__(self, n_out, kernel, stride, padding, dropout = True):
        super().__init__()
        self.up_conv = Sequential([
            ReLU(max_value=6),
            Conv2DTranspose(n_out, kernel, stride, padding),
            BatchNormalization()
        ])
        if dropout:
            self.up_conv.add(Dropout(0.5))

    def call(self, x):
        return self.up_conv(x)

class RE_Generator(Model):
    def __init__(self):
        super().__init__()

        # 36x60x3 -> 18x30x64
        self.conv1 = Conv2D(64, 4, 2, 'same')

        # 18x30x64 -> 9x15x128
        self.conv2 = Down(128, 4, 2, 'same')

        # 9x15x128 -> 4x7x256
        self.conv3 = Down(256, 3, 2, 'valid')

        # 4x7x256 -> 2x5x512
        self.conv4 = Down(512, 3, 1, 'valid')

        # 2x5x512 -> 1x4x512
        self.conv5 = Down(512, 2, 1, 'valid')

        # Up Sampling
        # 1x4x512 -> 2x5x512
        self.deconv5 = Up(512, 2, 1, 'valid', True)

        # 2x5x(512*2) -> 4x7x256
        self.deconv4 = Up(256, 3, 1, 'valid', True)

        # 4x7x(256*2) -> 9x15x128
        self.deconv3 = Up(128, 3, 2, 'valid', False)

        # 9x15x(128*2) -> 18x30x64
        self.deconv2 = Up(64, 4, 2, 'same', False)

        # 18x30x(64*2) -> 36x60x3
        self.deconv1 = Up(3, 4, 2, 'same', False)

        self.outlayer = Sequential([
            ReLU(),
            Conv2D(3, 1, 1, 'same')
        ])


    def call(self, x):
   
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)

        d4 = self.deconv5(c5)
        d4 = tf.concat([d4, c4], axis = 3)
        d3 = self.deconv4(d4)
        d3 = tf.concat([d3, c3], axis = 3)
        d2 = self.deconv3(d3)
        d2 = tf.concat([d2, c2], axis = 3)
        d1 = self.deconv2(d2)
        d1 = tf.concat([d1, c1], axis = 3)
        out = self.deconv1(d1)

        out = self.outlayer(out)

        return out



class Resnet_Block(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(64, 3, 1, 'same')
        self.relu1 = ReLU()
        self.conv2 = Conv2D(64, 3, 1, 'same')
        self.relu2 = ReLU()

    def call(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)

        x = tf.add(x, shortcut)
        out = self.relu2(x)

        return out


class Generator(Model):
    def __init__(self):
        super().__init__()
        # 36x60x3 -> 36x60x64
        self.conv1 = Conv2D(64, 3, 1, 'same', activation='relu')
        self.resblock = Resnet_Block()
        self.conv2 = Conv2D(3, 1, 1, 'same', activation='tanh')

    def call(self, x):
        x = self.conv1(x)

        for i in range(4):
            x = self.resblock(x)

        x = self.conv2(x)

        return x

class Discriminator(Model):
    def __init__(self):
        super().__init__()
        # 36x60x3 -> 18x30x96
        self.conv1 = Sequential([
            Conv2D(96, 3, 2, 'same'),
            BatchNormalization(),
            ReLU()
        ])

        # 18x30x96 -> 9x15x64
        self.conv2 = Sequential([
            Conv2D(64, 3, 2, 'same'),
            BatchNormalization()
        ])

        # 9x15x64 -> 4x7x64
        self.maxpool = MaxPool2D(3, 2, 'valid')

        # 4x7x64 -> 2x5x32
        self.conv3 = Sequential([
            Conv2D(32, 3, 1, 'valid'),
            BatchNormalization(),
            ReLU()
        ])

        # 2x5x32 -> 2x5x1
        self.conv4 = Sequential([
            Conv2D(1, 1, 1, 'same'),
            BatchNormalization()
        ])

        # 2x5x1 -> 1
        self.outlayer = Sequential([
            Flatten(),
            BatchNormalization(),
            Dense(1, 'sigmoid')
        ])


    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.outlayer(x)

        return x
