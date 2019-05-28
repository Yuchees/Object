#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variential auto encoder modes.
@author: Yu Che
"""
from keras import layers, Input, Model, metrics
from keras import backend as K


def encoder(params):
    input_matrix = Input(shape=params['input_shape'], name='encoder_input')
    x = layers.Conv1D(
        filters=params['number_of_filters'],
        kernel_size=params['kernel_size'],
        activation= params['activation'],
        name='encoder_conv0'
    )(input_matrix)
    if params['batch_norm']:
        x = layers.BatchNormalization(axis=-1, name='encoder_norm0')
    if params['num_conv'] > 1:
        for i in range(1, params['num_conv']):
            x = layers.Conv1D(
                filters=params['filters'],
                kernel_size=params['kernel_size'],
                activation=params['activation'],
                name='encoder_conv{}'.format(i)
            )(x)
            if params['batch_norm']:
                x = layers.BatchNormalization(
                    axis=-1,
                    name='encoder_norm{}'.format(i)
                )(x)
    shape_before_flattening = K.int_shape(x)
    x = layers.Flatten()(x)
    z_mean = layers.Dense(units=['hidden_dim'], name='encoder_mean')(x)
    z_log_var = layers.Dense(units=params['hidden_dim'], name='encoder_var')(x)
    encoder_model = Model(input_matrix, [z_mean, z_log_var], name='encoder')
    return encoder_model, shape_before_flattening


def sampling(args, params):
    z_mean, z_log_var = args
    epsilon = K.random_normal(
        shape=(K.shape(z_mean)[0], params['latent_dim']),
        mean=0.,stddev=1.)
    return z_mean + K.exp(z_log_var) * epsilon


def decoder(params):
    z_input = Input(shape=params['latent_dim'], name='decoder_input')
    z = layers.Dense()(z_input)
    z = layers.GRU(name='decoder_gru0')(z)
    z = layers.GRU(name='decoder_gru1')(z)
    z = layers.GRU(name='decoder_gru2')(z)
    decoder_model = Model(z_input, z, name='decoder')
    return decoder_model


class CustomVariationalLayer(layers.Layer):
    def vae_loss(self, x, z_decoded, args):
        z_mean, z_log_var = args
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        xent_loss = metrics.binary_crossentropy(x, z_decoded)
        kl_loss = -5e-4 * K.mean(
            1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x

