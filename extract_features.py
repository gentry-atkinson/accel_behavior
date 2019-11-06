import keras
from matplotlib import pyplot
import numpy as np

from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Dense, Reshape, Flatten
from keras.models import Model
from keras.optimizers import RMSprop

import keras.backend as K
from keras.layers import Conv2DTranspose, Lambda, BatchNormalizationl

DIRECTORIES = ["dws_1", "dws_2", "dws_11", "jog_9", "jog_16", "sit_5", "sit_13", "std_6", "std_14", "ups_3", "ups_4", "ups_12", "wlk_7", "wlk_8", "wlk_15"]
NUMBER_EPOCHS = 2



def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same'):
    #https://stackoverflow.com/questions/44061208/how-to-implement-the-conv1dtranspose-in-keras
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1),activation='linear', padding=padding)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x

def build_AE(train_set):
    #https://stackoverflow.com/questions/49552651/conv1d-convolutional-autoencoder-for-text-in-keras
    #https://missinglink.ai/guides/keras/keras-conv1d-working-1d-convolutional-neural-networks-keras/
    #https://medium.com/analytics-vidhya/building-a-convolutional-autoencoder-using-keras-using-conv2dtranspose-ca403c8d144e
    #https://blog.keras.io/building-autoencoders-in-keras.html

    #New data is time series 12d data

    print("Building input layer..................")
    inp = Input(train_set[0].shape)

    print("Building encoder.......................")
    enc = Reshape((12, 512))(inp)
    enc = Conv1D(filters=256, kernel_size=16, strides=8, activation='linear', padding='same')(enc)
    enc = MaxPooling1D(pool_size=4, padding='same')(enc)
    enc = Flatten()(enc)
    hidden = Dense(128, use_bias=True, activation='sigmoid')(enc)

    print("Building decoder......................")
    dec = hidden
    #dec = Dense(64, activation='relu')(dec)
    dec = Reshape((12, 128))(dec)
    dec = Conv1DTranspose(dec, filters=512, kernel_size=8, padding='same')
    #dec = UpSampling1D(size=512)(dec)
    dec = MaxPooling1D(pool_size=4, padding='same')(dec)
    #dec = BatchNormalization()(dec)
    dec = Flatten()(dec)

    print("Compiling model.......................")
    autoenc = Model(inp, dec)
    autoenc.compile(optimizer='RMSProp', loss='mean_squared_error', metrics=['acc'])

    autoenc.summary()
    return autoenc


def read_files():
    for i in range(FILES_PER_SET):
        a_name = "SET A/Z" + format(i+1, '03d') + ".txt"
        b_name = "SET B/O" + format(i+1, '03d') + ".txt"
        c_name = "SET C/N" + format(i+1, '03d') + ".TXT"
        d_name = "SET D/F" + format(i+1, '03d') + ".txt"
        e_name = "SET E/S" + format(i+1, '03d') + ".txt"

        a_file = open(a_name, "r")
        b_file = open(b_name, "r")
        c_file = open(c_name, "r")
        d_file = open(d_name, "r")
        e_file = open(e_name, "r")

        a_lines = a_file.readlines()
        b_lines = b_file.readlines()
        c_lines = c_file.readlines()
        d_lines = d_file.readlines()
        e_lines = e_file.readlines()

        j = 0;

        for a_val in a_lines:
            A[i, j] = int(a_val)
            j += 1

        j = 0;

        for b_val in b_lines:
            B[i, j] = int(b_val)
            j += 1

        j = 0;

        for c_val in c_lines:
            C[i, j] = int(c_val)
            j += 1

        j = 0;

        for d_val in d_lines:
            D[i, j] = int(d_val)
            j += 1

        j = 0;

        for e_val in e_lines:
            E[i, j] = int(e_val)
            j += 1

    print("A[0,0] " + str(A[0,0]))
    print("B[0,0] " + str(B[0,0]))
    print("C[0,0] " + str(C[0,0]))
    print("D[0,0] " + str(D[0,0]))
    print("E[0,0] " + str(E[0,0]))
