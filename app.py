import matplotlib.pyplot as plt
import tensorflow as tf

from flask import Flask, render_template
from tensorflow_examples.models.dcgan import dcgan

from models.dcgan import DCGAN
from models.target import MNISTConvTarget

NUM_IMAGES = 16
LATENT_DIM = 100
INPUT_SHAPE = (28, 28, 1)

GAN_WEIGHTS = '../weights/dcgan_mnist_20220114144039.h5'
TARGET_WEIGHTS = '../weights/mnist_conv_target_20220106120545.h5'


def load_gan(weights_path, latent_dim=LATENT_DIM, input_shape=INPUT_SHAPE):
    generator = dcgan.make_generator_model()
    generator.build((None, latent_dim))

    discriminator = dcgan.make_discriminator_model()
    discriminator.build((None, *input_shape))

    model = DCGAN(discriminator, generator, latent_dim=latent_dim)

    model.built = True
    model.load_weights(weights_path)

    return model


def load_target(weights_path):
    model = MNISTConvTarget()

    model.built = True
    model.load_weights(weights_path)

    return model


app = Flask(__name__)

gan = load_gan(GAN_WEIGHTS)
target = load_target(TARGET_WEIGHTS)


@app.route('/')
def main():
    return render_template('index.html')


@app.route('/images', methods=['POST'])
def images():
    image_path = 'static/images/generated.png'

    noise = tf.random.normal(shape=(NUM_IMAGES, LATENT_DIM))
    generated_images = gan.generator(noise, training=False)

    probs = target.predict(generated_images)
    max_probs = probs.max(axis=1)
    max_labels = probs.argmax(axis=1)

    for i in range(NUM_IMAGES):
        ax = plt.subplot(4, 4, i + 1)
        ax.set_title(f'{max_labels[i]} ({max_probs[i]:.4f})')

        plt.imshow(generated_images[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(image_path)

    return render_template('index.html', image_path=image_path)


if __name__ == '__main__':
    app.run()
