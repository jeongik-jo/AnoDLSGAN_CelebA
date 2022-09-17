import tensorflow as tf
import tensorflow.keras as kr
import HyperParameters as hp
import Models
import datetime
import time
import numpy as np
import Evaluate
import os
import matplotlib.pyplot as plt


def train(encoder: Models.Encoder, decoder: Models.Decoder, svm: Models.Svm, train_dataset, id_dataset, ood_datasets):
    @tf.function
    def _gan_train_step(encoder: kr.Model, encoder_optimizer: kr.optimizers.Optimizer, encoder_ema: tf.train.ExponentialMovingAverage,
                        decoder: kr.Model, decoder_optimizer: kr.optimizers.Optimizer, decoder_ema: tf.train.ExponentialMovingAverage,
                        latent_var_trace: tf.Variable, data, **kwargs):
        with tf.GradientTape(persistent=True) as tape:
            real_images = data['image']
            batch_size = real_images.shape[0]
            latent_vectors = hp.latent_dist_func(batch_size)

            if hp.is_dls:
                latent_scale_vector = tf.sqrt(
                    tf.cast(hp.latent_vector_dim, dtype='float32') * latent_var_trace / tf.reduce_sum(latent_var_trace))
            else:
                latent_scale_vector = tf.ones([hp.latent_vector_dim])

            with tf.GradientTape() as reg_tape:
                reg_tape.watch(real_images)
                real_adv_values, _ = encoder(real_images)
            reg_losses = tf.reduce_sum(tf.square(reg_tape.gradient(real_adv_values, real_images)), axis=[1, 2, 3])

            fake_images = decoder(latent_vectors * latent_scale_vector[tf.newaxis])
            fake_adv_values, rec_latent_vectors = encoder(fake_images)

            enc_losses = tf.reduce_mean(
                tf.square((rec_latent_vectors - latent_vectors) * latent_scale_vector[tf.newaxis]), axis=-1)

            dis_adv_losses = tf.nn.softplus(-real_adv_values) + tf.nn.softplus(fake_adv_values)
            gen_adv_losses = tf.nn.softplus(-fake_adv_values)

            dis_loss = tf.reduce_mean(dis_adv_losses + hp.enc_weight * enc_losses + hp.reg_weight * reg_losses)
            gen_loss = tf.reduce_mean(gen_adv_losses + hp.enc_weight * enc_losses)

        decoder_optimizer.apply_gradients(
            zip(tape.gradient(gen_loss, decoder.trainable_variables),
                decoder.trainable_variables)
        )
        encoder_optimizer.apply_gradients(
            zip(tape.gradient(dis_loss, encoder.trainable_variables),
                encoder.trainable_variables)
        )

        del tape

        decoder_ema.apply(decoder.trainable_variables)
        encoder_ema.apply(encoder.trainable_variables)

        latent_var_trace.assign(latent_var_trace * hp.latent_var_decay_rate +
                                tf.reduce_mean(tf.square(rec_latent_vectors), axis=0) * (1.0 - hp.latent_var_decay_rate))

        results = {'real_adv_values': real_adv_values, 'fake_adv_values': fake_adv_values,
                   'enc_losses': enc_losses, 'reg_losses': reg_losses}

        return results

    @tf.function
    def _autoencoder_train_step(encoder: kr.Model, encoder_optimizer: kr.optimizers.Optimizer, encoder_ema: tf.train.ExponentialMovingAverage,
                                decoder: kr.Model, decoder_optimizer: kr.optimizers.Optimizer, decoder_ema: tf.train.ExponentialMovingAverage,
                                data, **kwargs):
        with tf.GradientTape(persistent=True) as tape:
            real_images = data['image']
            rec_images = decoder(encoder(real_images)[1])

            rec_losses = tf.reduce_mean(tf.square(rec_images - real_images), axis=[1, 2, 3])
            rec_loss = tf.reduce_mean(rec_losses)
        decoder_optimizer.apply_gradients(
            zip(tape.gradient(rec_loss, decoder.trainable_variables),
                decoder.trainable_variables)
        )
        encoder_optimizer.apply_gradients(
            zip(tape.gradient(rec_loss, encoder.trainable_variables),
                encoder.trainable_variables)
        )

        del tape

        decoder_ema.apply(decoder.trainable_variables)
        encoder_ema.apply(encoder.trainable_variables)

        results = {'rec_losses': rec_losses}

        return results

    @tf.function
    def _classifier_train_step(encoder: kr.Model, encoder_optimizer: kr.optimizers.Optimizer, encoder_ema: tf.train.ExponentialMovingAverage,
                               svm: kr.Model, svm_optimizer: kr.optimizers.Optimizer, svm_ema: tf.train.ExponentialMovingAverage,
                               data, **kwargs):
        with tf.GradientTape(persistent=True) as tape:
            real_images = data['image']
            real_labels = data['label']
            predict_logits = svm(tf.nn.relu(encoder(real_images)[1]))
            ce_losses = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(real_labels, predict_logits), axis=-1)
            ce_loss = tf.reduce_mean(ce_losses)

        encoder_optimizer.apply_gradients(
            zip(tape.gradient(ce_loss, encoder.trainable_variables),
                encoder.trainable_variables)
        )
        svm_optimizer.apply_gradients(
            zip(tape.gradient(ce_loss, svm.trainable_variables),
                svm.trainable_variables)
        )

        del tape

        svm_ema.apply(svm.trainable_variables)
        encoder_ema.apply(encoder.trainable_variables)

        results = {'ce_losses': ce_losses}
        return results

    print('\ntraining...')
    evaluate_results_sets = []
    total_start = time.time()
    for epoch in range(hp.epochs):
        print(datetime.datetime.now())
        print('epoch', epoch)
        epoch_start = time.time()

        train_results = {}

        for data in train_dataset:
            args = {'encoder': encoder.model, 'encoder_optimizer': encoder.optimizer, 'encoder_ema': encoder.ema,
                    'decoder': decoder.model, 'decoder_optimizer': decoder.optimizer, 'decoder_ema': decoder.ema,
                    'svm': svm.model, 'svm_optimizer': svm.optimizer, 'svm_ema': svm.ema,
                    'latent_var_trace': decoder.latent_var_trace, 'data': data}
            if hp.train_gan:
                batch_results = _gan_train_step(**args)
            elif hp.train_autoencoder:
                batch_results = _autoencoder_train_step(**args)
            elif hp.train_classifier:
                batch_results = _classifier_train_step(**args)
            else:
                raise AssertionError

            for key in batch_results:
                try:
                    train_results[key].append(batch_results[key])
                except KeyError:
                    train_results[key] = [batch_results[key]]

        temp_results = {}
        for key in train_results:
            mean, variance = tf.nn.moments(tf.concat(train_results[key], axis=0), axes=0)
            temp_results[key + '_mean'] = mean
            temp_results[key + '_variance'] = variance
        train_results = temp_results

        for key in train_results:
            print('%-30s:' % key, '%13.6f' % np.array(train_results[key]))
        print('epoch time: ', time.time() - epoch_start, '\n')

        encoder.save()
        decoder.save()
        svm.save()

        encoder.to_ema()
        decoder.to_ema()
        svm.to_ema()

        decoder.save_images(encoder.model, id_dataset, ood_datasets, epoch)
        print('\nevaluating...')
        print(datetime.datetime.now())
        start = time.time()
        evaluate_results = Evaluate.evaluate(encoder, decoder, svm, id_dataset, ood_datasets, epoch)
        evaluate_results_sets.append(evaluate_results)
        for key in evaluate_results:
            print('%-50s:' % key, '%13.6f' % evaluate_results[key])
        print('evaluate time: ', time.time() - start, '\n')
        encoder.to_train()
        decoder.to_train()
        svm.to_train()

    temp_results = {}
    for evaluate_results in evaluate_results_sets:
        for key in evaluate_results:
            try:
                temp_results[key].append(evaluate_results[key])
            except KeyError:
                temp_results[key] = [evaluate_results[key]]
    evaluate_results = temp_results

    if not os.path.exists('results/figures'):
        os.makedirs('results/figures')
    file_name = 'results/figures/'
    if hp.train_gan:
        if hp.is_dls:
            file_name += 'DLSGAN'
        else:
            file_name += 'InfoGAN'
    elif hp.train_autoencoder:
        file_name += 'Autoencoder'
    elif hp.train_classifier:
        file_name += 'Classifier'
    else:
        raise AssertionError

    for key in evaluate_results:
        np.savetxt(file_name + '_%s.txt' % key, np.array(evaluate_results[key]), fmt='%f')
        plt.title(key)
        plt.xlabel('Epochs')
        plt.ylabel(key)
        plt.plot([(i + 1) for i in range(len(evaluate_results[key]))], evaluate_results[key])
        plt.savefig(file_name + '_%s.png' % key)
        plt.clf()

    print('total time: ', time.time() - total_start, '\n')

