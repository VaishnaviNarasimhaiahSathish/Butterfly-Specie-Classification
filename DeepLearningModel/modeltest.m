% Load the trained model
loaded = load('alexnetcheck.mat');
net = loaded.net;

% Use the trained network to classify a single test image
img = imread('/MATLAB Drive/deeplearning/DatasetComplete/COMMONROSE/50.jpg');
img = imresize(img, [227, 227]);
predictedLabel = classify(net, img);
fprintf("Predicted Label: %s\n", predictedLabel);
[out,score]=classify(net,img)
figure,
imshow(img)
title(string(out))

% Use the trained network to classify all images in the test set
augmentedTestSet = augmentedImageDatastore(inputSize, testSet, 'ColorPreprocessing', 'gray2rgb');
predictedLabels = classify(net, augmentedTestSet);

% Calculate the accuracy of the network
accuracy = mean(predictedLabels == testSet.Labels);
fprintf("Accuracy = %.2f%%\n", accuracy*100);

% Compute the confusion matrix
figure;
plotconfusion(testSet.Labels, predictedLabels);
