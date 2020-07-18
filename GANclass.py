from keras.layers import Dense, Input, Flatten, LeakyReLU, Conv2D, UpSampling2D
from keras.layers import BatchNormalization, Activation, Dropout, Reshape
from keras.models import Model, Sequential
import keras.backend as K
from keras.initializers import RandomNormal
from keras import optimizers
from keras import losses
from keras import utils
import numpy as np



class GAN():
    def __init__(self,
                 input_dim,
                 discriminator_conv_filters,
                 discriminator_conv_kernel_size,
                 discriminator_conv_strides,
                 discriminator_batch_normal_momentum,
                 discriminator_activation,
                 discriminator_dropout_rate,
                 discriminator_learning_rate,
                 generator_conv_filters,
                 generator_conv_kernel_size,
                 generator_conv_strides,
                 generator_batch_normal_momentum,
                 generator_activation,
                 generator_dropout_rate,
                 generator_learning_rate,
                 generator_initial_dense_size,
                 generator_upscale,
                 optimizer,
                 z_dim,
                 WGAN):

        self.input_dim = input_dim
        self.discriminator_conv_filters = discriminator_conv_filters
        self.discriminator_conv_strides = discriminator_conv_strides
        self.discriminator_conv_kernel_size = discriminator_conv_kernel_size
        self.discriminator_batch_normal_momentum = discriminator_batch_normal_momentum
        self.discriminator_activation = discriminator_activation
        self.discriminator_dropout_rate = discriminator_dropout_rate
        self.discriminator_learning_rate = discriminator_learning_rate

        self.generator_conv_filters = generator_conv_filters
        self.generator_conv_kernel_size = generator_conv_kernel_size
        self.generator_conv_strides = generator_conv_strides
        self.generator_batch_normal_momentum = generator_batch_normal_momentum
        self.generator_activation = generator_activation
        self.generator_dropout_rate = generator_dropout_rate
        self.generator_learning_rate = generator_learning_rate
        self.generator_initial_dense_size = generator_initial_dense_size
        self.generator_upscale = generator_upscale
        self.optimizer = optimizer
        self.z_dim = z_dim
        self.n_discriminator_conv_layers = len(self.discriminator_conv_filters)
        self.n_generator_conv_layers = len(self.generator_conv_filters)

        self.weight_init = RandomNormal(mean = 0., stddev = .02)
        self.d_losses = []
        self.g_losses = []
        self.WGAN = WGAN



    def build_discriminator(self):
        discriminator_input = Input(shape = self.input_dim, name = 'discriminator_input')
        x = discriminator_input

        for i in range(self.n_discriminator_conv_layers):
            x = Conv2D(
                filters = self.discriminator_conv_filters[i],
                kernel_size = self.discriminator_conv_kernel_size[i],
                strides = self.discriminator_conv_strides[i],
                padding = 'same',
                name = 'discriminator_conv_layer_' + str(i)
            )(x)

            if self.discriminator_batch_normal_momentum and i>0:
                x = BatchNormalization(momentum = self.discriminator_batch_normal_momentum)(x)

            x = Activation(self.discriminator_activation)(x)

            if self.discriminator_dropout_rate:
                x = Dropout(self.discriminator_dropout_rate)(x)

        x = Flatten()(x)

        if self.WGAN == False:
            discriminator_output = Dense(1, activation = 'sigmoid', kernel_initializer = self.weight_init)(x)

        else:
            discriminator_output = Dense(1, kernel_initializer=self.weight_init)(x)

        self.discriminator = Model(discriminator_input, discriminator_output)



    def build_generator(self):
        self.generator_input = Input(shape = (self.z_dim,), name = 'generator_input')
        x = self.generator_input

        x = Dense(np.prod(self.generator_initial_dense_size))(x)
        x = Reshape(self.generator_initial_dense_size)(x)

        if self.generator_batch_normal_momentum:
            x = BatchNormalization(momentum = self.generator_batch_normal_momentum)(x)
        if self.generator_dropout_rate:
            x = Dropout(self.generator_dropout_rate)(x)

        for i in range(self.n_generator_conv_layers):
            if self.generator_upscale[i]==2:
                x = UpSampling2D()(x)
            else:
                pass

            x = Conv2D(
                filters = self.generator_conv_filters[i],
                kernel_size = self.generator_conv_kernel_size[i],
                padding = 'same',
                name = 'generator_conv_' + str(i)
            )(x)

            if i< self.n_generator_conv_layers - 1:
                if self.generator_batch_normal_momentum:
                    x = BatchNormalization(momentum = self.generator_batch_normal_momentum)(x)
                x = Activation(self.generator_activation)(x)
                if self.generator_dropout_rate:
                    x = Dropout(self.generator_dropout_rate)(x)
            else:
                if self.generator_batch_normal_momentum:
                    x = BatchNormalization(momentum = self.generator_batch_normal_momentum)(x)
                x = Activation('tanh')(x)

            generator_output = x
            self.generator = Model(self.generator_input, generator_output)

    def compile(self):

        if self.WGAN == False:
            self.discriminator.compile(
                optimizer = optimizers.RMSprop(self.discriminator_learning_rate),
                loss = losses.binary_crossentropy,
                metrics = ['accuracy']
            )
        else:
            def wasserstein(y_true, y_pred):
                return -K.mean(y_true*y_pred)

            self.discriminator.compile(
                optimizer=optimizers.RMSprop(self.discriminator_learning_rate),
                loss=wasserstein,
            )

        self.discriminator.trainable = False
        model_input = Input(shape = (self.z_dim,), name = 'model_input')
        model_output = self.discriminator(self.generator(model_input))
        self.model = Model(model_input, model_output)

        if self.WGAN == False:
            self.model.compile(
                optimizer = optimizers.RMSprop(self.generator_learning_rate),
                loss = losses.binary_crossentropy,
                metrics = ['accuracy']
            )
        else:
            self.model.compile(
                optimizer=optimizers.RMSprop(self.generator_learning_rate),
                loss=wasserstein,
            )

        self.discriminator.trainable = True


    def train_discriminator(self, x_train, batch_size, clip_threshold):
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))


        idx = np.random.randint(0,x_train.shape[0], batch_size)
        true_img = x_train[idx]

        if self.WGAN==False:
            d_loss_real, d_acc_real = self.discriminator.train_on_batch(true_img, valid)
        else:
            d_loss_real = self.discriminator.train_on_batch(true_img, valid)


        noise = np.random.normal(0., 1, (batch_size, self.z_dim))
        fake_img = self.generator.predict(noise)

        if self.WGAN==False:
            clip_threshold = None
            d_loss_fake, d_acc_fake = self.discriminator.train_on_batch(fake_img, fake)

            d_loss = .5*(d_loss_real + d_loss_fake)
            d_acc = .5*(d_acc_real + d_acc_fake)

            return [d_loss, d_acc, d_loss_real, d_loss_fake, d_acc_real, d_acc_fake]

        else:
            fake = -np.ones((batch_size, 1))
            d_loss_fake = self.discriminator.train_on_batch(fake_img, fake)

            for l in self.discriminator.layers:
                weights = l.get_weights()
                weights = [np.clip(w, -w*clip_threshold, w*clip_threshold) for w in weights]
                l.set_weights(weights)

            d_loss = .5 * (d_loss_real + d_loss_fake)

            return [d_loss, d_loss_real, d_loss_fake]


    def train_generator(self, batch_size):
        valid = np.ones((batch_size, 1))

        noise = np.random.normal(0., 1., (batch_size, self.z_dim))
        return self.model.train_on_batch(noise, valid) #앞에서 model을 input 을 generator_input 으로 정의했으므로, noise 만 넣으면


    def train(self, x_train, epochs, batch_size, clip_threshold, n_critic):
        for i in range(epochs):
            for j in range(n_critic):
                d = self.train_discriminator(x_train, batch_size, clip_threshold)
            g = self.train_generator(batch_size)

            self.d_losses.append(d)
            self.g_losses.append(g)

        return self.d_losses, selfg_losses



