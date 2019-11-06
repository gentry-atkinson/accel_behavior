import keras
from matplotlib import pyplot
import numpy as np

from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Dense, Reshape, Flatten
from keras.models import Model
from keras.optimizers import RMSprop

import keras.backend as K
from keras.layers import Conv2DTranspose, Lambda, BatchNormalization

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
    #splitting data 80/20 eventually
    all_data = np.zeros((360, 370, 12), dtype='float')
    all_labels = np.zeros((360), dtype='int')

    directory_num = 0

    print("Loading from: ", len(DIRECTORIES), " directories")

    for directory in DIRECTORIES:
        print("Reading ", directory)
        filepath = "motionsense-dataset/A_DeviceMotion_data/" + directory + "/"
        for i in range (1, 25):
            #print("Subject: ", i, " has offset ", directory_num, ". Index is ", i+directory_num-1)
            filename = filepath + "sub_" + str(i) +".csv"
            f = open(filename, 'r')
            f_lines = f.readlines()
            for j in range(1, 371):
                values = f_lines[j].split(',')
                for k in range(1, 13):
                    all_data[i+directory_num-1][j-1][k-1] = values[k]
            if "dws" in directory:
                all_labels[i+directory_num-1] = 0
            elif "jog" in directory:
                all_labels[i+directory_num-1] = 1
            elif "sit" in directory:
                all_labels[i+directory_num-1] = 2
            elif "std" in directory:
                all_labels[i+directory_num-1] = 3
            elif "ups" in directory:
                all_labels[i+directory_num-1] = 4
            elif "wlk" in directory:
                all_labels[i+directory_num-1] = 5
            else:
                print("Bad directory name in read_files()")
        directory_num += 24


def main():
    read_files()

if __name__ == "__main__":
    main()
