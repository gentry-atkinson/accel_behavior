cluster_matrix = zeros(6, 6);
clusters = kmeans(trainingdata, 6);

for i=1:288
    c = clusters(i);
    l = rawtrainlabels1d(i);
    cluster_matrix(l, c) =  cluster_matrix(l, c) + 1;
end

label_counts = [58, 58, 58, 58, 58, 58; 
    38, 38, 38, 38, 38, 38; 
    39, 39, 39, 39, 39, 39; 
    38, 38, 38, 38, 38, 38; 
    58, 58, 58, 58, 58, 58; 
    57, 57, 57, 57, 57, 57];
cluster_percentages = cluster_matrix ./ label_counts;

range = 1:1:370;
figure(1);
plot(range, rawtraindata1d(61, :), range, decodeddata(288, :));
figure(2);
plot(range, rawtraindata1d(99, :), range, decodeddata(288, :));

%Label 1: Downstairs, 58 samples, start at 1
%Label 2: Jog, 38 samples, starts at 59
%Label 3: Sit, 39 samples, starts at 97
%Label 4: Stand, 38 samples, starts at 136
%Label 5: Upstairs, 58 samples, starts at 174
%Label 6: Walking, 57 samples, starts at 232