cluster_matrix = zeros(6, 6);
clusters = kmeans(trainingdata, 6);

for i=1:22925
    c = clusters(i);
    l = rawtrainlabels1d(i);
    cluster_matrix(l, c) =  cluster_matrix(l, c) + 1;
end



c_c = zeros(6,1);
l_c = zeros(6,1);

for i = 1:22925
   c_c(clusters(i)) = c_c(clusters(i)) + 1;
   l_c(rawtrainlabels1d(i)) = l_c(rawtrainlabels1d(i)) + 1;
end

for i = 2:22925
   if rawtrainlabels1d(i) ~= rawtrainlabels1d(i-1);
      fprintf("Label %i starts at %i \n", rawtrainlabels1d(i), i);
   end     
end

cluster_count_array = [c_c(1), c_c(2), c_c(3), c_c(4), c_c(5), c_c(6);
    c_c(1), c_c(2), c_c(3), c_c(4), c_c(5), c_c(6);
    c_c(1), c_c(2), c_c(3), c_c(4), c_c(5), c_c(6);
    c_c(1), c_c(2), c_c(3), c_c(4), c_c(5), c_c(6);
    c_c(1), c_c(2), c_c(3), c_c(4), c_c(5), c_c(6);
    c_c(1), c_c(2), c_c(3), c_c(4), c_c(5), c_c(6);];

label_counts = [l_c(1), l_c(2), l_c(3), l_c(4), l_c(5), l_c(6);
    l_c(1), l_c(2), l_c(3), l_c(4), l_c(5), l_c(6);
    l_c(1), l_c(2), l_c(3), l_c(4), l_c(5), l_c(6);
    l_c(1), l_c(2), l_c(3), l_c(4), l_c(5), l_c(6);
    l_c(1), l_c(2), l_c(3), l_c(4), l_c(5), l_c(6);
    l_c(1), l_c(2), l_c(3), l_c(4), l_c(5), l_c(6);];

%The percentage of each cluster composed of a label
label_percentage = cluster_matrix ./ cluster_count_array;

%The percentage of each label in a cluster
cluster_percentages = cluster_matrix ./ label_counts;

range = 1:1:50;
figure(1);
title('Downstairs')
plot(range, rawtraindata1d(1, :), range, decodeddata(1, :));
figure(2);
title('Jog')
plot(range, rawtraindata1d(2125, :), range, decodeddata(2125, :));
figure(3);
title('Sit')
plot(range, rawtraindata1d(4297, :), range, decodeddata(4297, :));
figure(4);
title('Stand')
plot(range, rawtraindata1d(9812, :), range, decodeddata(9812, :));
figure(5);
title('Upstairs')
plot(range, rawtraindata1d(14793, :), range, decodeddata(14793, :));
figure(6);
title('Walk')
plot(range, rawtraindata1d(17331, :), range, decodeddata(17331, :));


%Label 1: Downstairs, 58 samples, start at 1
%Label 2: Jog, 38 samples, starts at 59
%Label 3: Sit, 39 samples, starts at 97
%Label 4: Stand, 38 samples, starts at 136
%Label 5: Upstairs, 58 samples, starts at 174
%Label 6: Walking, 57 samples, starts at 232