cluster_matrix = zeros(6, 6);

for i=1:288
    c = clusters(i);
    l = rawtrainlabels1d(i)+1;
    cluster_matrix(l, c) =  cluster_matrix(l, c) + 1;
end

label_counts = 
cluster_percentages = cluster_matrix / 288;
