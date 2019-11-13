import numpy as np
import math

DIRECTORIES = ["dws_1", "dws_2", "dws_11", "jog_9", "jog_16", "sit_5", "sit_13", "std_6", "std_14", "ups_3", "ups_4", "ups_12", "wlk_7", "wlk_8", "wlk_15"]

def resultant_vector(x_axis, y_axis, z_axis):
    return math.sqrt((x_axis * x_axis) + (y_axis * y_axis) + (z_axis * z_axis))

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
        #print(filename)
        f = open(filename, 'r')
        f_lines = f.readlines()
        for j in range(1, 371):
            values = f_lines[j].split(',')
            for k in range(1, 13):
                all_data[i+directory_num-1][j-1][k-1] = values[k]
                #print(all_data[i+directory_num-1][j-1][k-1])
        if "dws" in directory:
            all_labels[i+directory_num-1] = 1
        elif "jog" in directory:
            all_labels[i+directory_num-1] = 2
        elif "sit" in directory:
            all_labels[i+directory_num-1] = 3
        elif "std" in directory:
            all_labels[i+directory_num-1] = 4
        elif "ups" in directory:
            all_labels[i+directory_num-1] = 5
        elif "wlk" in directory:
            all_labels[i+directory_num-1] = 6
        else:
            print("Bad directory name in read_files()")
    directory_num += 24

train_data = np.zeros((288, 370), dtype='float')
train_labels = np.zeros((288), dtype='int')
test_data = np.zeros((72, 370), dtype='float')
test_labels = np.zeros((72), dtype='int')

test_counter = 0;
train_counter = 0;

for i in range(360):
    for j in range(370):
        #print ("line ", i, " value ", j)
        if i%5 == 4:
            test_data[test_counter][j] = resultant_vector(all_data[i][j][9], all_data[i][j][10], all_data[i][j][11])
            test_labels[test_counter] = all_labels[i]
            #print("test", test_data[test_counter][j])
        else:
            train_data[train_counter][j] = resultant_vector(all_data[i][j][9], all_data[i][j][10], all_data[i][j][11])
            train_labels[train_counter] = all_labels[i]
            #print("train", train_data[train_counter][j])
    if i%5 == 4:
        test_counter += 1
    else:
        train_counter += 1

print(test_counter, " test samples recorded")
print(train_counter, " train counters recorded")
print(len(train_data[0]), " samples per segment")
print("train_data[287][369]", train_data[287][369])

np.savetxt("raw_train_data_1d.csv", train_data, delimiter=',')
np.savetxt("raw_train_labels_1d.csv", train_labels, delimiter=',')
np.savetxt("raw_test_data_1d.csv", test_data, delimiter=',')
np.savetxt("raw_test_labels_1d.csv", test_labels, delimiter=',')
