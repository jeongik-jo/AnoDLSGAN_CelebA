import os
import tensorflow as tf
import tensorflow.keras as kr
import HyperParameters as hp
import tensorflow_datasets as tfds
import numpy as np


@tf.function
def _crop_and_normalize(data):
    images = data['image'][:, 45:-45, 25:-25, :]
    images = tf.image.resize(tf.cast(images, 'float32') / 127.5 - 1.0, [hp.image_resolution, hp.image_resolution])
    labels = tf.stack([tf.cast(data['attributes'][attribute], 'float32') for attribute in hp.train_attributes], axis=1)
    return {'image': images, 'label': labels}


@tf.function
def _resize_and_normalize(data):
    images = tf.image.resize(tf.cast(data['image'], 'float32') / 127.5 - 1.0, size=[hp.image_resolution, hp.image_resolution])
    return {'image': images}


def load_id_dataset():
    dataset = tfds.load('celeb_a')
    train_dataset = dataset['train']
    id_dataset = dataset['test']

    if hp.train_data_size != -1:
        train_dataset = train_dataset.take(hp.train_data_size)
    if hp.test_data_size != -1:
        id_dataset = id_dataset.take(hp.test_data_size)

    train_dataset = train_dataset.shuffle(1000).batch(hp.batch_size, drop_remainder=True).map(
        _crop_and_normalize, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    id_dataset = id_dataset.batch(hp.batch_size, drop_remainder=True).map(
        _crop_and_normalize, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    return train_dataset, id_dataset


def load_ood_datasets(id_dataset):
    ood_datasets = {}
    save_samples = []
    for id_data in id_dataset.take(1):
        id_samples = id_data['image'][:hp.save_image_size]
        save_samples.append(np.vstack(id_samples))

    for ood_dataset_name in hp.ood_datasets:
        ood_datasets[ood_dataset_name] = tfds.load(ood_dataset_name)['train'].map(
            _resize_and_normalize, num_parallel_calls=tf.data.AUTOTUNE).batch(
            hp.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

        for ood_data in ood_datasets[ood_dataset_name].take(1):
            for ood_intensity in hp.ood_intensities:
                ood_samples = ood_data['image'][:hp.save_image_size]
                interpolate_images = ood_samples * ood_intensity + id_samples * (1 - ood_intensity)
                save_samples.append(np.vstack(interpolate_images))

    if not os.path.exists('results/samples'):
        os.makedirs('results/samples')
    kr.preprocessing.image.save_img(path='results/samples/ood_samples.png', x=np.hstack(save_samples))

    return ood_datasets
