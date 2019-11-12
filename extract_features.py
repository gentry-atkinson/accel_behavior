#### TODO:
#       1)only take raw accel input
#       2)visualize decoded dataset

##############################################
#Author:    Gentry Atkinson
#Org:       Texas State University
#Date:      6 November 2019
##############################################

import keras
from matplotlib import pyplot
import numpy as np

from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Dense, Reshape, Flatten
from keras.models import Model
from keras.optimizers import RMSprop

import keras.backend as K
from keras.layers import Conv2DTranspose, Lambda, BatchNormalization

DIRECTORIES = ["dws_1", "dws_2", "dws_11", "jog_9", "jog_16", "sit_5", "sit_13", "std_6", "std_14", "ups_3", "ups_4", "ups_12", "wlk_7", "wlk_8", "wlk_15"]
NUMBER_EPOCHS = 5



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
    enc = Reshape((370, 1))(inp)
    enc = Conv1D(filters=256, kernel_size=16, strides=8, activation='linear', padding='same')(enc)
    enc = MaxPooling1D(pool_size=8, padding='same')(enc)
    enc = Flatten()(enc)
    hidden = Dense(128, use_bias=True, activation='sigmoid')(enc)

    print("Building decoder......................")
    dec = hidden
    dec = Dense(256, activation='sigmoid')(dec)
    #print(dec.shape)
    dec = Reshape((256, 1))(dec)
    dec = Conv1DTranspose(dec, filters=370, kernel_size=8, padding='same')
    #dec = UpSampling1D(size=512)(dec)
    dec = MaxPooling1D(pool_size=512, padding='same')(dec)
    dec = Reshape((370, 1))(dec)
    dec = Flatten()(dec)
    #dec = BatchNormalization()(dec)

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

    train_data = np.zeros((288, 370, 12), dtype='float')
    train_labels = np.zeros((288), dtype='int')
    test_data = np.zeros((72, 370, 12), dtype='float')
    test_labels = np.zeros((72), dtype='int')

    test_counter = 0;
    train_counter = 0;

    for i in range(360):
        for j in range(370):
            for k in range(12):
                if i%5 == 4:
                    test_data[test_counter][j][k] = all_data[i][j][k]
                    test_labels[test_counter] = all_labels[i]
                else:
                    train_data[test_counter][j][k] = all_data[i][j][k]
                    train_labels[train_counter] = all_labels[i]
        if i%5 == 4:
            test_counter += 1
        else:
            train_counter += 1

    print(test_counter, " test samples recorded")
    print(train_counter, " train counters recorded")
    return train_data, train_labels, test_data, test_labels

def read_files_1d():
    train_data = np.loadtxt("raw_train_data_1d.csv", delimiter=',')
    train_labels = np.loadtxt("raw_train_labels_1d.csv", delimiter=',')
    test_data = np.loadtxt("raw_test_data_1d.csv", delimiter=',')
    test_labels = np.loadtxt("raw_test_labels_1d.csv", delimiter=',')
    return train_data, train_labels, test_data, test_labels

def train_AE(autoenc, train_set):
    print("-----------------Training Autoencoder-----------------")
    print("Passed " + str(len(train_set)) + " segments.")
    print("Segments have length " + str(len(train_set[0])))

    autoenc.fit(train_set, train_set, epochs=NUMBER_EPOCHS)
    return autoenc

def test_AE(model, test_set):
    print("----------------Testing AutoEncoder------------------")
    accuracy = model.evaluate(test_set, test_set, verbose=1)
    print("Loss & accuracy: ", accuracy)

def trim_decoder(autoenc):
    print("Removing decoder from autoencoder....")
    o = autoenc.layers[-9].output
    encoder = Model(input=autoenc.input, output=[o])
    encoder.summary()
    return encoder

def write_features(train_file, test_file, encoder, train_set, test_set):
    print("Writing features to " + train_file)
    features = encoder.predict_on_batch(train_set)
    print("Writing ", len(features), " features of size ", len(features[0]))
    np.savetxt(train_file, features, delimiter=',')

    print("Writing features to " + test_file)
    features = encoder.predict_on_batch(test_set)
    print("Writing ", len(features), " features of size ", len(features[0]))
    np.savetxt(test_file, features, delimiter=',')

def write_labels(train_labels, test_labels, train_label_set, test_label_set):
    print("Writilng labels to " + train_labels)
    np.savetxt(train_labels, train_label_set)

    print("Writilng labels to " + test_labels)
    np.savetxt(test_labels, test_label_set)

def write_decoded_data(autoenc, train_data):
    print("Writing decoded values as CSV")
    decoded_data = autoenc.predict_on_batch(train_data)
    np.savetxt("decoded_data.csv", decoded_data)


def main():
    train_data, train_labels, test_data, test_labels = read_files_1d()
    autoenc = build_AE(train_data)
    autoenc = train_AE(autoenc, train_data)
    write_decoded_data(autoenc, train_data)
    test_AE(autoenc, test_data)
    encoder = trim_decoder(autoenc)

    write_features("training_data.csv", "testing_data.csv", encoder, train_data, test_data)
    write_labels("training_labels.csv", "testing_labels.csv", train_labels, test_labels)


if __name__ == "__main__":
    main()
