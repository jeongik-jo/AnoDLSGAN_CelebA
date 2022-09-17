import tensorflow as tf
import tensorflow.keras as kr
import Layers
import os
import HyperParameters as hp
import numpy as np


class Decoder(object):
    def build_model(self):
        latent_vector = kr.Input([hp.latent_vector_dim])
        fake_image = Layers.Decoder()(latent_vector)
        return kr.Model(latent_vector, fake_image)

    def __init__(self):
        self.model = self.build_model()
        self.optimizer = kr.optimizers.Adam(learning_rate=hp.learning_rate, beta_1=0.0, beta_2=0.99)
        self.latent_var_trace = tf.Variable(tf.ones([hp.latent_vector_dim]))
        self.ema = tf.train.ExponentialMovingAverage(decay=hp.weight_ema_decay_rate)
        self.ema.apply(self.model.trainable_variables)
        if hp.train_gan:
            self.save_latent_vectors_sets = [hp.latent_dist_func(hp.save_image_size) for _ in range(hp.save_image_size)]

    def save(self):
        if not os.path.exists('models'):
            os.makedirs('models')
        self.model.save_weights('models/decoder.h5')
        np.save('models/latent_var_trace.npy', self.latent_var_trace)

    def load(self):
        self.model.load_weights('models/decoder.h5')
        self.latent_var_trace = tf.Variable(np.load('models/latent_var_trace.npy'))

    def to_ema(self):
        self.train_weights = [tf.constant(weight) for weight in self.model.trainable_variables]
        for weight in self.model.trainable_variables:
            weight.assign(self.ema.average(weight))

    def to_train(self):
        for ema_weight, train_weight in zip(self.model.trainable_variables, self.train_weights):
            ema_weight.assign(train_weight)

    def save_images(self, encoder, id_dataset, ood_datasets, epoch):
        if hp.train_gan or hp.train_autoencoder:
            if hp.train_gan:
                if hp.is_dls:
                    latent_scale_vector = tf.sqrt(tf.cast(hp.latent_vector_dim, dtype='float32') *
                                              self.latent_var_trace / tf.reduce_sum(self.latent_var_trace))
                else:
                    latent_scale_vector = tf.ones([hp.latent_vector_dim])

                if not os.path.exists('results/samples/fake_images'):
                    os.makedirs('results/samples/fake_images')
                images = []
                for i in range(hp.save_image_size):
                    fake_images = self.model(self.save_latent_vectors_sets[i] * latent_scale_vector[tf.newaxis])
                    images.append(np.hstack(fake_images))

                kr.preprocessing.image.save_img(path='results/samples/fake_images/fake_%d.png' % epoch,
                                                x=tf.clip_by_value(np.vstack(images), clip_value_min=-1,
                                                                   clip_value_max=1))

            else:
                latent_scale_vector = tf.ones([hp.latent_vector_dim])

            if not os.path.exists('results/samples/rec_images'):
                os.makedirs('results/samples/rec_images')

            id_rec_images = []
            for id_data in id_dataset.take(hp.save_image_size // 2):
                id_images = id_data['image'][:hp.save_image_size]
                _, rec_latent_vectors = encoder(id_images)
                rec_images = self.model(rec_latent_vectors * latent_scale_vector[tf.newaxis])

                id_rec_images.append(np.vstack(id_images))
                id_rec_images.append(np.vstack(rec_images))
                id_rec_images.append(tf.ones([np.vstack(id_images).shape[0], 5, 3]))

            kr.preprocessing.image.save_img(path='results/samples/rec_images/id_rec_%d.png' % epoch,
                                            x=tf.clip_by_value(np.hstack(id_rec_images), clip_value_min=-1,
                                                               clip_value_max=1))

            for key in ood_datasets:
                ood_rec_images = []
                for ood_data in ood_datasets[key].take(hp.save_image_size // 2):
                    for ood_intensity in hp.ood_intensities:
                        ood_images = ood_data['image'][:hp.save_image_size]
                        interpolate_images = ood_images * ood_intensity + id_images * (1 - ood_intensity)
                        _, rec_latent_vectors = encoder(interpolate_images)
                        rec_images = self.model(rec_latent_vectors * latent_scale_vector[tf.newaxis])

                        ood_rec_images.append(np.vstack(interpolate_images))
                        ood_rec_images.append(np.vstack(rec_images))
                        ood_rec_images.append(tf.ones([np.vstack(interpolate_images).shape[0], 5, 3]))

                kr.preprocessing.image.save_img(path='results/samples/rec_images/ood_rec_%s_%d.png' % (key, epoch),
                                                x=tf.clip_by_value(np.hstack(ood_rec_images), clip_value_min=-1,
                                                                   clip_value_max=1))

class Encoder(object):
    def build_model(self):
        input_image = kr.Input([hp.image_resolution, hp.image_resolution, 3])
        adv_value, feature_vector = Layers.Encoder()(input_image)
        return kr.Model(input_image, [adv_value, feature_vector])

    def __init__(self):
        self.model = self.build_model()
        self.optimizer = kr.optimizers.Adam(learning_rate=hp.learning_rate, beta_1=0.0, beta_2=0.99)
        self.ema = tf.train.ExponentialMovingAverage(decay=hp.weight_ema_decay_rate)
        self.ema.apply(self.model.trainable_variables)

    def save(self):
        if not os.path.exists('models'):
            os.makedirs('models')
        self.model.save_weights('models/encoder.h5')

    def load(self):
        self.model.load_weights('models/encoder.h5')

    def to_ema(self):
        self.train_weights = [tf.constant(weight) for weight in self.model.trainable_variables]
        for weight in self.model.trainable_variables:
            weight.assign(self.ema.average(weight))

    def to_train(self):
        for ema_weight, train_weight in zip(self.model.trainable_variables, self.train_weights):
            ema_weight.assign(train_weight)


class Svm(object):
    def build_model(self):
        input_image = kr.Input([hp.latent_vector_dim])
        predict_logit = Layers.Svm()(input_image)
        return kr.Model(input_image, predict_logit)

    def __init__(self):
        self.model = self.build_model()
        self.optimizer = kr.optimizers.Adam(learning_rate=hp.learning_rate, beta_1=0.0, beta_2=0.99)
        self.ema = tf.train.ExponentialMovingAverage(decay=hp.weight_ema_decay_rate)
        self.ema.apply(self.model.trainable_variables)

    def save(self):
        if not os.path.exists('models'):
            os.makedirs('models')
        self.model.save_weights('models/svm.h5')

    def load(self):
        self.model.load_weights('models/svm.h5')

    def to_ema(self):
        self.train_weights = [tf.constant(weight) for weight in self.model.trainable_variables]
        for weight in self.model.trainable_variables:
            weight.assign(self.ema.average(weight))

    def to_train(self):
        for ema_weight, train_weight in zip(self.model.trainable_variables, self.train_weights):
            ema_weight.assign(train_weight)
