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
