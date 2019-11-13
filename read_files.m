decodeddata_table = readtable("decoded_data.csv", 'HeaderLines', 0, 'ReadVariableNames', false);
rawtraindata1d_table = readtable("raw_train_data_1d.csv", 'HeaderLines', 0, 'ReadVariableNames', false);
rawtrainlabels1d_table = readtable("raw_train_labels_1d.csv", 'HeaderLines', 0, 'ReadVariableNames', false);
trainingdata_table = readtable("training_data.csv", 'HeaderLines', 0, 'ReadVariableNames', false);

decodeddata = decodeddata_table{:,:};
rawtraindata1d = rawtraindata1d_table{:,:};
rawtrainlabels1d = rawtrainlabels1d_table{:,:};
trainingdata = trainingdata_table{:,:};