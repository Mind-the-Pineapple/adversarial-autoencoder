"""
Deterministic supervised adversarial autoencoder.

 We are using:
    - Gaussian distribution as prior distribution.
    - Convolutional layers.
    - Cyclic learning rate.
"""
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patches as mpatches
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path.cwd()

# -------------------------------------------------------------------------------------------------------------
# Set random seed
random_seed = 42
tf.random.set_seed(random_seed)
np.random.seed(random_seed)

# -------------------------------------------------------------------------------------------------------------
output_dir = PROJECT_ROOT / 'outputs'
output_dir.mkdir(exist_ok=True)

experiment_dir = output_dir / 'supervised_aae_deterministic_convolutional'
experiment_dir.mkdir(exist_ok=True)

latent_space_dir = experiment_dir / 'latent_space'
latent_space_dir.mkdir(exist_ok=True)

reconstruction_dir = experiment_dir / 'reconstruction'
reconstruction_dir.mkdir(exist_ok=True)

style_dir = experiment_dir / 'style'
style_dir.mkdir(exist_ok=True)

# -------------------------------------------------------------------------------------------------------------
# Loading data
print("Loading data...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# -------------------------------------------------------------------------------------------------------------
# Create the dataset iterator
batch_size = 256
train_buf = 60000

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=train_buf)
train_dataset = train_dataset.batch(batch_size)

# -------------------------------------------------------------------------------------------------------------
# Create models
n_labels = 10


def make_encoder_model(z_size):
    inputs = tf.keras.layers.Input(shape=(28, 28, 1))

    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same')(inputs)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    z = tf.keras.layers.Conv2D(filters=z_size, kernel_size=3, strides=2, padding='same')(x)

    model = tf.keras.Model(inputs=inputs, outputs=z)
    return model


def make_decoder_model(z_size):
    encoded = tf.keras.Input(shape=(1, 1, z_size + n_labels))

    x = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(encoded)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)

    reconstruction = tf.keras.layers.Conv2D(filters=1, kernel_size=3, activation='sigmoid', padding='same')(x)
    decoder = tf.keras.Model(inputs=encoded, outputs=reconstruction)
    return decoder


def make_discriminator_model(z_size):
    encoded = tf.keras.Input(shape=(z_size,))
    x = tf.keras.layers.Dense(128)(encoded)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    prediction = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=encoded, outputs=prediction)
    return model


z_dim = 2
encoder = make_encoder_model(z_dim)
decoder = make_decoder_model(z_dim)
discriminator = make_discriminator_model(z_dim)

# -------------------------------------------------------------------------------------------------------------
# Define loss functions
ae_loss_weight = 1.
gen_loss_weight = 1.
dc_loss_weight = 1.

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
mse = tf.keras.losses.MeanSquaredError()
accuracy = tf.keras.metrics.BinaryAccuracy()


def autoencoder_loss(inputs, reconstruction, loss_weight):
    return loss_weight * mse(inputs, reconstruction)


def discriminator_loss(real_output, fake_output, loss_weight):
    loss_real = cross_entropy(tf.ones_like(real_output), real_output)
    loss_fake = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return loss_weight * (loss_fake + loss_real)


def generator_loss(fake_output, loss_weight):
    return loss_weight * cross_entropy(tf.ones_like(fake_output), fake_output)


# -------------------------------------------------------------------------------------------------------------
# Define cyclic learning rate
base_lr = 0.00025
max_lr = 0.0025

n_samples = 60000
step_size = 2 * np.ceil(n_samples / batch_size)
global_step = 0

# -------------------------------------------------------------------------------------------------------------
# Define optimizers
ae_optimizer = tf.keras.optimizers.Adam(lr=base_lr)
dc_optimizer = tf.keras.optimizers.Adam(lr=base_lr)
gen_optimizer = tf.keras.optimizers.Adam(lr=base_lr)


# -------------------------------------------------------------------------------------------------------------
# Training function
@tf.function
def train_step(batch_x, batch_y):
    # -------------------------------------------------------------------------------------------------------------
    # Autoencoder
    with tf.GradientTape() as ae_tape:
        encoder_output = encoder(batch_x, training=True)
        labels = tf.reshape(tf.one_hot(batch_y, n_labels), [batch_y.shape[0], 1, 1, n_labels])
        decoder_output = decoder(tf.concat([encoder_output, labels], axis=-1), training=True)

        # Autoencoder loss
        ae_loss = autoencoder_loss(batch_x, decoder_output, ae_loss_weight)

    ae_grads = ae_tape.gradient(ae_loss, encoder.trainable_variables + decoder.trainable_variables)
    ae_optimizer.apply_gradients(zip(ae_grads, encoder.trainable_variables + decoder.trainable_variables))

    # -------------------------------------------------------------------------------------------------------------
    # Discriminator
    with tf.GradientTape() as dc_tape:
        real_distribution = tf.random.normal([batch_x.shape[0], 1, 1, z_dim], mean=0.0, stddev=1.0)
        encoder_output = encoder(batch_x, training=True)

        dc_real = discriminator(real_distribution, training=True)
        dc_fake = discriminator(encoder_output, training=True)

        # Discriminator Loss
        dc_loss = discriminator_loss(dc_real, dc_fake, dc_loss_weight)

        # Discriminator Acc
        dc_acc = accuracy(tf.concat([tf.ones_like(dc_real), tf.zeros_like(dc_fake)], axis=0),
                          tf.concat([dc_real, dc_fake], axis=0))

    dc_grads = dc_tape.gradient(dc_loss, discriminator.trainable_variables)
    dc_optimizer.apply_gradients(zip(dc_grads, discriminator.trainable_variables))

    # -------------------------------------------------------------------------------------------------------------
    # Generator (Encoder)
    with tf.GradientTape() as gen_tape:
        encoder_output = encoder(batch_x, training=True)
        dc_fake = discriminator(encoder_output, training=True)

        # Generator loss
        gen_loss = generator_loss(dc_fake, gen_loss_weight)

    gen_grads = gen_tape.gradient(gen_loss, encoder.trainable_variables)
    gen_optimizer.apply_gradients(zip(gen_grads, encoder.trainable_variables))

    return ae_loss, dc_loss, dc_acc, gen_loss


# -------------------------------------------------------------------------------------------------------------
# Training loop
n_epochs = 350
for epoch in range(n_epochs):
    start = time.time()

    # Learning rate schedule
    if epoch in [60, 100, 300]:
        base_lr = base_lr / 2
        max_lr = max_lr / 2
        step_size = step_size / 2

        print('learning rate changed!')

    epoch_ae_loss_avg = tf.metrics.Mean()
    epoch_dc_loss_avg = tf.metrics.Mean()
    epoch_dc_acc_avg = tf.metrics.Mean()
    epoch_gen_loss_avg = tf.metrics.Mean()

    for batch, (batch_x, batch_y) in enumerate(train_dataset):
        # -------------------------------------------------------------------------------------------------------------
        # Calculate cyclic learning rate
        global_step = global_step + 1
        cycle = np.floor(1 + global_step / (2 * step_size))
        x_lr = np.abs(global_step / step_size - 2 * cycle + 1)
        clr = base_lr + (max_lr - base_lr) * max(0, 1 - x_lr)
        ae_optimizer.lr = clr
        dc_optimizer.lr = clr
        gen_optimizer.lr = clr

        ae_loss, dc_loss, dc_acc, gen_loss = train_step(batch_x, batch_y)

        epoch_ae_loss_avg(ae_loss)
        epoch_dc_loss_avg(dc_loss)
        epoch_dc_acc_avg(dc_acc)
        epoch_gen_loss_avg(gen_loss)

    epoch_time = time.time() - start
    print('{:4d}: TIME: {:.2f} ETA: {:.2f} AE_LOSS: {:.4f} DC_LOSS: {:.4f} DC_ACC: {:.4f} GEN_LOSS: {:.4f}' \
          .format(epoch, epoch_time,
                  epoch_time * (n_epochs - epoch),
                  epoch_ae_loss_avg.result(),
                  epoch_dc_loss_avg.result(),
                  epoch_dc_acc_avg.result(),
                  epoch_gen_loss_avg.result()))

    # -------------------------------------------------------------------------------------------------------------
    if epoch % 10 == 0:
        # Latent Space
        x_test_encoded = encoder(x_test, training=False)
        label_list = list(y_test)

        fig = plt.figure()
        classes = set(label_list)
        colormap = plt.cm.rainbow(np.linspace(0, 1, len(classes)))
        kwargs = {'alpha': 0.8, 'c': [colormap[i] for i in label_list]}
        ax = plt.subplot(111, aspect='equal')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        handles = [mpatches.Circle((0, 0), label=class_, color=colormap[i])
                   for i, class_ in enumerate(classes)]
        ax.legend(handles=handles, shadow=True, bbox_to_anchor=(1.05, 0.45),
                  fancybox=True, loc='center left')
        plt.scatter(x_test_encoded[:, 0, 0, 0], x_test_encoded[:, 0, 0, 1], s=2, **kwargs)
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])

        plt.savefig(latent_space_dir / ('epoch_%d.png' % epoch))
        plt.close('all')

        # Reconstruction
        n_digits = 20  # how many digits we will display
        test_labels = tf.reshape(tf.one_hot(y_test[:n_digits], n_labels), [n_digits, 1, 1, n_labels])
        x_test_decoded = decoder(
            tf.concat([x_test_encoded[:n_digits], test_labels], axis=-1), training=False)

        x_test_decoded = np.reshape(x_test_decoded, [-1, 28, 28]) * 255
        fig = plt.figure(figsize=(20, 4))
        for i in range(n_digits):
            # display original
            ax = plt.subplot(2, n_digits, i + 1)
            plt.imshow(x_test[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n_digits, i + 1 + n_digits)
            plt.imshow(x_test_decoded[i])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.savefig(reconstruction_dir / ('epoch_%d.png' % epoch))
        plt.close('all')

        # Conditioned Sampling
        nx, ny = 10, 10
        random_inputs = np.random.randn(10, z_dim)
        sample_y = np.identity(10)
        plt.subplot()
        gs = gridspec.GridSpec(nx, ny, hspace=0.05, wspace=0.05)
        i = 0
        for r in random_inputs:
            for t in sample_y:
                r = np.reshape(r, (1, 1, 1, z_dim))
                t = np.reshape(t, (1, 1, 1, n_labels))
                dec_input = np.concatenate((r, t), -1)
                x = decoder(dec_input.astype('float32'), training=False).numpy()
                ax = plt.subplot(gs[i])
                i += 1
                img = np.array(x.tolist()).reshape(28, 28)
                ax.imshow(img, cmap='gray')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_aspect('auto')

        plt.savefig(style_dir / ('epoch_%d.png' % epoch))
        plt.close()
