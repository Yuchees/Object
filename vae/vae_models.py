#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational auto encoder models and properties prediction
@author: Yu Che
"""
from keras import layers, Input, Model, metrics
from keras import backend as K


def encoder(params):
    encoder_input = Input(shape=params['input_shape'], name='encoder_input')
    x = layers.Conv1D(
        filters=params['conv_filters'][0],
        kernel_size=params['conv_kernel_size'][0],
        activation=params['conv_activation'][0],
        padding=params['conv_padding'][0],
        name='encoder_conv0'
    )(encoder_input)
    if params['conv_batch_norm']:
        x = layers.BatchNormalization(axis=-1, name='encoder_norm0')(x)
    if params['num_conv'] > 1:
        for i in range(1, params['num_conv']):
            x = layers.Conv1D(
                filters=params['conv_filters'][i],
                kernel_size=params['conv_kernel_size'][i],
                activation=params['conv_activation'][i],
                padding=params['conv_padding'][i],
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
        name='decoder_middle'
    )(decoder_input)
    z = layers.RepeatVector(n=params['input_shape'][0])(z)
    output = layers.GRU(
        units=params['gru_dim'][0],
        activation=params['gru_activation'][0],
        return_sequences=True,
        name='decoder_gru0'
    )(z)
    if params['num_gru'] > 1:
        for i in range(1, params['num_gru']):
            output = layers.GRU(
                units=params['gru_dim'][i],
                activation=params['gru_activation'][i],
                return_sequences=True,
                name='decoder_gru{}'.format(i)
            )(output)
    decoder_model = Model(decoder_input, output, name='decoder')
    return decoder_model


def vae_loss(z_mean, z_log_var, x, y):
    reconstruction_loss = metrics.binary_crossentropy(
        K.flatten(x),
        K.flatten(y)
    )
    kl_loss = -0.5 * K.mean(
        1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1
    )
    return K.mean(reconstruction_loss + kl_loss)


def property_predication(params):
    prop_input = Input(shape=params['latent_dim'], name='prediction_input')
    x = layers.Dense(
        units=params['prop_pred_dim'][0],
        activation=params['prop_pred_activation'][0],
        name='prop_pred0'
    )(prop_input)
    if params['num_prop_layer'] > 1:
        for i in range(1, params['num_prop_layer']):
            x = layers.Dense(
                units=params['prop_pred_dim'][i],
                activation=params['prop_pred_activation'][i],
                name='prop_pred{}'.format(i)
            )(x)
    prop_pred_model = Model(prop_input, x, name='prop_pred')
    return prop_pred_model
