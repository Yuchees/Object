#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational auto encoder models
@author: Yu Che
"""
from keras import layers, Input, Model, metrics
from keras import backend as K


def encoder(params):
    encoder_input = Input(shape=params['input_shape'], name='encoder_input')
    x = layers.Conv1D(
        filters=params['conv_filters'],
        kernel_size=params['conv_kernel_size'],
        activation=params['conv_activation'],
        padding=params['conv_padding'],
        name='encoder_conv0'
    )(encoder_input)
    if params['conv_batch_norm']:
        x = layers.BatchNormalization(axis=-1, name='encoder_norm0')(x)
    if params['num_conv'] > 1:
        for i in range(1, params['num_conv']):
            x = layers.Conv1D(
                filters=params['conv_filters'],
                kernel_size=params['conv_kernel_size'],
                activation=params['conv_activation'],
                padding=params['conv_padding'],
                name='encoder_conv{}'.format(i)
            )(x)
            if params['conv_batch_norm']:
                x = layers.BatchNormalization(
                    axis=-1,
                    name='encoder_norm{}'.format(i)
                )(x)
    x = layers.Flatten()(x)
    z_mean = layers.Dense(
        units=params['latent_dim'], name='encoder_mean'
    )(x)
    z_log_var = layers.Dense(
        units=params['latent_dim'], name='encoder_var'
    )(x)

    def sampling(args):
        # Reparameterization
        mean, log_var = args
        epsilon = K.random_normal(
            shape=(K.shape(mean)[0], params['latent_dim']), mean=0., stddev=1.
        )
        return mean + K.exp(0.5 * log_var) * epsilon
    z_sampling = layers.Lambda(sampling, name='vae_layer')([z_mean, z_log_var])
    encoder_model = Model(
        encoder_input, [z_mean, z_log_var, z_sampling],
        name='encoder'
    )
    return encoder_model


def decoder(params):
    decoder_input = Input(shape=(params['latent_dim'],), name='decoder_input')
    z = layers.Dense(
        units=params['middle_dim'],
        activation=params['middle_activation'],
        name='decoder_dense'
    )(decoder_input)
    z = layers.RepeatVector(n=params['input_shape'][0])(z)
    output = layers.GRU(
        units=params['gru_dim'],
        activation=params['gru_activation'],
        return_sequences=True,
        name='decoder_gru0'
    )(z)
    if params['num_gru'] > 1:
        for i in range(1, params['num_gru']):
            output = layers.GRU(
                units=params['gru_dim'],
                activation=params['gru_activation'],
                return_sequences=True,
                name='decoder_gru{}'.format(i)
            )(output)
    decoder_model = Model(decoder_input, output, name='decoder')
    return decoder_model


def vae_loss(z_mean, z_log_var, x, y):
    reconstruction_loss = metrics.binary_crossentropy(x, y)
    kl_loss = -0.5 * K.mean(
        1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1
    )
    return K.mean(reconstruction_loss + kl_loss)
