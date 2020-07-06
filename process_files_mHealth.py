import numpy as np
import math
from sklearn.preprocessing import normalize

DIRECTORIES = ["MHEALTHDATASET"]
SAMPLE_RATE_IN_HERTZ = 50
SAMPLE_LENGTH_IN_SECONDS = 1
SAMPLE_LENGTH = SAMPLE_RATE_IN_HERTZ * SAMPLE_LENGTH_IN_SECONDS

array_size = 1000
num_samples = 0

def resultant_vector(x_axis, y_axis, z_axis):
    return math.sqrt((x_axis * x_axis) + (y_axis * y_axis) + (z_axis * z_axis))

def grow_data_array(all_data, array_size):
    new_array_size = array_size * 2
    new_array = np.zeros((new_array_size, SAMPLE_LENGTH, 3), dtype='float')
    for i in range(array_size):
        for j in range(SAMPLE_LENGTH):
            for k in range(3):
                new_array[i][j][k] = all_data[i][j][k]
    return new_array, new_array_size

def grow_label_array(all_labels, array_size):
    new_array = np.zeros((array_size), dtype='int')
    for i in range(len(all_labels)):
        new_array[i] = all_labels[i]
    return new_array

def shrink_data_array(all_data, num_samples):
    new_array = np.zeros((num_samples, SAMPLE_LENGTH, 3), dtype='float')
    for i in range(num_samples):
        for j in range(SAMPLE_LENGTH):
            for k in range(3):
                new_array[i][j][k] = all_data[i][j][k]
    return new_array

def shrink_label_array(all_labels, num_samples):
    new_array = np.zeros((num_samples), dtype='int')
    for i in range(num_samples):
        new_array[i] = all_labels[i]
    return new_array

#splitting data 80/20 eventually
all_data = np.zeros((array_size, 50, 3), dtype='float')
all_labels = np.zeros((array_size), dtype='int')

directory_num = 0

print("Loading from: ", len(DIRECTORIES), " directories")

for directory in DIRECTORIES:
    print("Reading ", directory)
    filepath = "extra_datasets/" + directory + "/"
    for i in range (1, 10):
        print("Subject: ", i, " has offset ", directory_num, ". Index is ", i+directory_num-1)

        filename = filepath + "mHealth_subject" + str(i) +".log"
        #print(filename)
        f = open(filename, 'r')
        f_lines = f.readlines()
        print(len(f_lines), " lines read from ", filename)
        c = 0;
        #firstline = True
        for line in f_lines:
            #if(firstline):
            #    firstline = False
            #    continue
            values = line.split( )
            #print(len(values), " columns in line")
            #acceleration from the right-lower-arm sensor X,Y,Z
            all_data[num_samples][c][0] = values[14]
            all_data[num_samples][c][1] = values[15]
            all_data[num_samples][c][2] = values[16]
            c = c+1
            if(c==SAMPLE_LENGTH-1):
                c = 0
            #    if "dws" in directory:
            #        all_labels[num_samples] = 1
            #    elif "jog" in directory:
            #        all_labels[num_samples] = 2
            #    elif "sit" in directory:
            #        all_labels[num_samples] = 3
            #    elif "std" in directory:
            #        all_labels[num_samples] = 4
            #    elif "ups" in directory:
            #        all_labels[num_samples] = 5
            #    elif "wlk" in directory:
            #        all_labels[num_samples] = 6
            #    else:
            #        print("Bad directory name in read_files()")
                all_labels[num_samples] = values[23]
                num_samples = num_samples + 1
                if(num_samples == array_size):
                    all_data, array_size = grow_data_array(all_data, array_size)
                    all_labels = grow_label_array(all_labels, array_size)
    directory_num += 24

all_data = shrink_data_array(all_data, num_samples)
all_labels = shrink_label_array(all_labels, num_samples)

import math
num_train = math.ceil(0.8*num_samples)
num_test = num_samples - num_train

train_data = np.zeros((num_train, SAMPLE_LENGTH), dtype='float')
train_labels = np.zeros((num_train), dtype='int')
test_data = np.zeros((num_test, SAMPLE_LENGTH), dtype='float')
test_labels = np.zeros((num_test), dtype='int')

test_counter = 0;
train_counter = 0;

for i in range(num_samples):
    for j in range(SAMPLE_LENGTH):
        #print ("line ", i, " value ", j)
        if i%5 == 4:
            test_data[test_counter][j] = resultant_vector(all_data[i][j][0], all_data[i][j][1], all_data[i][j][2])
            test_labels[test_counter] = all_labels[i]
            #print("test", test_data[test_counter][j])
        else:
            train_data[train_counter][j] = resultant_vector(all_data[i][j][0], all_data[i][j][1], all_data[i][j][2])
            train_labels[train_counter] = all_labels[i]
            #print("train", train_data[train_counter][j])
    if i%5 == 4:
        test_counter += 1
    else:
        train_counter += 1

train_data = normalize(train_data, norm='l2', copy=False)
test_data = normalize(test_data, norm='l2', copy=False)

print(test_counter, " test samples recorded")
print(train_counter, " train counters recorded")
print(len(train_data[0]), " samples per segment")
print("last train_data val ", train_data[num_train-1][SAMPLE_LENGTH-1])

np.savetxt("mHealth_train_data_1d.csv", train_data, delimiter=',')
np.savetxt("mHealth_train_labels_1d.csv", train_labels, delimiter=',')
np.savetxt("mHealth_test_data_1d.csv", test_data, delimiter=',')
np.savetxt("mHealth_test_labels_1d.csv", test_labels, delimiter=',')
