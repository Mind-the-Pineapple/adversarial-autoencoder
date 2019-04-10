"""

"""
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patches as mpatches
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path('/media/kcl_1/HDD/PycharmProjects/adversarial-autoencoder')

# Set random seed
tf.random.set_seed(42)
np.random.seed(42)

output_dir = PROJECT_ROOT / 'outputs'
output_dir.mkdir(exist_ok=True)

experiment_dir = output_dir / 'supervised'
experiment_dir.mkdir(exist_ok=True)

latent_space_dir = experiment_dir / 'latent_space'
latent_space_dir.mkdir(exist_ok=True)

reconstruction_dir = experiment_dir / 'reconstruction'
reconstruction_dir.mkdir(exist_ok=True)

style_dir = experiment_dir / 'style'
style_dir.mkdir(exist_ok=True)

# Loading data
print("Loading data...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# flatten the dataset
x_train = x_train.reshape((-1, 28 * 28))
x_test = x_test.reshape((-1, 28 * 28))

# CREATE MODEL
image_size = 784
batch_size = 256
h_dim = 1000
z_dim = 2
n_labels = 10

ae_loss_weight = 1.
gen_loss_weight = 1.
dc_loss_weight = 1.

learning_rate = 0.001
beta1 = 0.9


def make_encoder_model():
    inputs = tf.keras.Input(shape=(image_size,))
    x = tf.keras.layers.Dense(h_dim)(inputs)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(h_dim)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    encoded = tf.keras.layers.Dense(z_dim)(x)
    model = tf.keras.Model(inputs=inputs, outputs=encoded)
    return model


def make_decoder_model():
    encoded = tf.keras.Input(shape=(z_dim + n_labels,))
    x = tf.keras.layers.Dense(h_dim)(encoded)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(h_dim)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    reconstruction = tf.keras.layers.Dense(image_size, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=encoded, outputs=reconstruction)
    return model


def make_discriminator_model():
    encoded = tf.keras.Input(shape=(z_dim,))
    x = tf.keras.layers.Dense(h_dim)(encoded)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(h_dim)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    prediction = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=encoded, outputs=prediction)
    return model


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
mse = tf.keras.losses.MeanSquaredError()
accuracy = tf.keras.metrics.BinaryAccuracy()


def autoencoder_loss(inputs, reconstruction, loss_weigth):
    return loss_weigth * mse(inputs, reconstruction)


def discriminator_loss(real_output, generated_output, loss_weight):
    loss_real = cross_entropy(tf.ones_like(real_output), real_output)
    loss_fake = cross_entropy(tf.zeros_like(generated_output), generated_output)

    return loss_weight * (loss_fake + loss_real)


def generator_loss(generated_output, loss_weight):
    return loss_weight * cross_entropy(tf.ones_like(generated_output), generated_output)


encoder = make_encoder_model()
decoder = make_decoder_model()
discriminator = make_discriminator_model()

ae_optimizer = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=beta1)
dc_optimizer = tf.keras.optimizers.Adam(lr=learning_rate / 5, beta_1=beta1)
gen_optimizer = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=beta1)

batch_size = 256
# create the database iterator
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024)
train_dataset = train_dataset.batch(batch_size)

@tf.function
def train_step(batch_x, batch_y):
    real_distribution = tf.random.normal([batch_size, z_dim], mean=0.0, stddev=1.0)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as dc_tape, tf.GradientTape() as ae_tape:
        encoder_output = encoder(batch_x, training=True)
        decoder_output = decoder(tf.concat([encoder_output, tf.one_hot(batch_y, n_labels)], axis=1), training=True)

        d_real = discriminator(real_distribution, training=True)
        d_fake = discriminator(encoder_output, training=True)

        # Autoencoder loss
        ae_loss = autoencoder_loss(batch_x, decoder_output, ae_loss_weight)

        # Discrimminator Loss
        dc_loss = discriminator_loss(d_real, d_fake, dc_loss_weight)

        # Generator loss
        gen_loss = generator_loss(d_fake, gen_loss_weight)

        # Discriminator Acc
        dc_acc = (accuracy(tf.ones_like(d_real), d_real) + accuracy(tf.zeros_like(d_fake), d_fake)) / 2

    ae_grads = ae_tape.gradient(ae_loss, encoder.trainable_variables + decoder.trainable_variables)
    ae_optimizer.apply_gradients(zip(ae_grads, encoder.trainable_variables + decoder.trainable_variables))

    dc_grads = dc_tape.gradient(dc_loss, discriminator.trainable_variables)
    dc_optimizer.apply_gradients(zip(dc_grads, discriminator.trainable_variables))

    gen_grads = gen_tape.gradient(gen_loss, encoder.trainable_variables)
    gen_optimizer.apply_gradients(zip(gen_grads, encoder.trainable_variables))

    return ae_loss, dc_loss, gen_loss, dc_acc


n_epochs = 200
for epoch in range(n_epochs):
    start = time.time()

    epoch_ae_loss_avg = tf.metrics.Mean()
    epoch_dc_loss_avg = tf.metrics.Mean()
    epoch_dc_acc_avg = tf.metrics.Mean()
    epoch_gen_loss_avg = tf.metrics.Mean()

    for batch, (batch_x, batch_y) in enumerate(train_dataset):
        ae_loss, dc_loss, gen_loss, dc_acc = train_step(batch_x, batch_y)

        epoch_ae_loss_avg(ae_loss)
        epoch_dc_loss_avg(dc_loss)
        epoch_gen_loss_avg(gen_loss)
        epoch_dc_acc_avg(dc_acc)

    epoch_time = time.time() - start
    print('{:4d}: TIME: {:.2f} ETA: {:.2f} AE_LOSS: {:.4f} DC_LOSS: {:.4f} DC_ACC: {:.4f} GEN_LOSS: {:.4f}' \
          .format(epoch, epoch_time,
                  epoch_time * (n_epochs - epoch),
                  epoch_ae_loss_avg.result(),
                  epoch_dc_loss_avg.result(),
                  epoch_dc_acc_avg.result(),
                  epoch_gen_loss_avg.result()))

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
        plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], s=2, **kwargs)
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])

        plt.savefig(latent_space_dir / ('epoch_%d.png' % epoch))
        plt.close('all')

        # Reconstruction
        n_digits = 20  # how many digits we will display
        x_test_decoded = decoder(
            tf.concat([x_test_encoded[:n_digits], tf.one_hot(y_test[:n_digits], n_labels)], axis=1), training=False)

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
                r = np.reshape(r, (1, z_dim))
                t = np.reshape(t, (1, n_labels))
                dec_input = np.concatenate((r, t), 1)
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
