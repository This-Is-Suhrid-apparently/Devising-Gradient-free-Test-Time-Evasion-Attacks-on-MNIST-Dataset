clear all; close all; clc;

load net;

d = load('data.mat');


trainX  = d.trainX;
testX = d.testX;
trainY = d.trainY;
testY = d.testY;
%plot one training image
imshow(trainX(:,:,:,1));

mini_testX = testX(:,:,:,1:100);
mini_testY = testY(:,1:100);

adv_labels = randperm(10,10) - 1;

adv_labels_map = zeros(size(mini_testY));

for i = 1:numel(mini_testY)
    adv_labels_map(i) = adv_labels(mini_testY(i)+1);
end

[z,acc_list] = zoo_stoch_image(net, mini_testX, mini_testY, adv_labels_map);

% z = zoo_stoch(net, testX, testY, adv_labels);
% [z,acc_list] = zoo_stoch(net, mini_testX, mini_testY, adv_labels_map);


%this is the network outputs when the inputs are the test images
outputs = predict(net, testX);


% %this will give the predicted labels 
% predLabelsTest = net.classify(testX);
% %this gives the test accuracy
% accuracy = sum(predLabelsTest == categorical(transpose(testY))) / numel(testY);

% predLabelsTest = net.classify(testX_corr);
% %calc acc
% accuracy = sum(predLabelsTest == categorical(transpose(testY))) / numel(testY);
% disp(accuracy)


%plot a sample test image 
% imshow(testX(:,:,:,1));   


