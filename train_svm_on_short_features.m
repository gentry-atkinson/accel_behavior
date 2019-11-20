raw_model = fitcecoc(decodeddata, rawtrainlabels1d)
feature_model = fitcecoc(trainingdata, rawtrainlabels1d)

raw_cv = crossval(raw_model)
feat_cv = crossval(feature_model)

raw_error = kfoldLoss(raw_cv)
feat_error = kfoldLoss(feat_cv)

fprintf("Train on raw data: %d percent loss\n", raw_error*100)
fprintf("Train on extracted features: %d percent loss\n", feat_error*100)