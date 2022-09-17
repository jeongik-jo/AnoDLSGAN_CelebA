import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf

learning_rate = 0.001
weight_ema_decay_rate = 0.999

image_resolution = 128
train_attributes = ['Bangs', 'Black_Hair', 'Male', 'Smiling', 'Young']

ood_datasets = ['coil100', 'deep_weeds', 'stl10', 'cars196', 'cassava',
                'colorectal_histology', 'malaria', 'stanford_dogs', 'stanford_online_products']

ood_intensities = [0.5, 1.0]
latent_vector_dim = 512

train_data_size = -1
test_data_size = 10000

train_gan = True
train_autoencoder = False
train_classifier = False


#GAN
if train_gan:
    is_dls = True
    reg_weight = 1.0
    enc_weight = 1.0
    latent_var_decay_rate = 0.999
    latent_dist_func = lambda batch_size: tf.random.normal([batch_size, latent_vector_dim])

#Energy
elif train_classifier:
    temperatures = [1.0, 10.0]
    react_ps = [0.85, 0.90, 0.95, 1.0]

epochs = 10

batch_size = 16
load_model = False
save_image_size = 8





