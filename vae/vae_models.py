#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Yu Che
"""
import keras
from keras import layers, Input, Model, metrics
from keras import backend as K


def encoder(params):
    input_matrix = Input(shape=params['input_shape'], name='encoder_input')
    x = layers.Conv1D(name='encoder_conv0')(input_matrix)
    x = layers.Conv1D(name='encoder_conv1')(x)
    x = layers.Conv1D(name='encoder_conv2')(x)
    shape_before_flattening = K.int_shape(x)
    x = layers.Flatten()(x)
    z_mean = layers.Dense()(x)
    z_log_var = layers.Dense()(x)
    encoder_model = Model(input_matrix, [z_mean, z_log_var], name='encoder')
    return encoder_model, shape_before_flattening


def sampling(args, params):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean, )[0], params['latent_dim']), mean=0.,
                              stddev=1.)
    return z_mean + K.exp(z_log_var) * epsilon


def decoder(params):
    z_input = Input(shape=params['latent_dim'], name='decoder_input')
    z = layers.Dense()(z_input)
    z = layers.GRU()(z)
    z = layers.GRU()(z)
    z = layers.GRU()(z)
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

