%% run data attacks and get performance metrics on MIST test set
clear all; close all;
% addpath('utils')
load net;
%% test parameters
% Optimization routine to use. Set this to 'ADAM' or 'Newton'
% algorithm ="ADAM";
algorithm ="Newton";
%% run attack on each test set image, and store results
% array of possible target labels
classLabels = 1:10;
numTestIms = 10;

% count number of successful attacks 
successTrials = false(numTestIms, 1);

% array storing number of queries us d for each image
numQueriesTrials = zeros(size(successTrials));
% store images with added adversarial noise
imsAdversarial = zeros([size(testX,1),size(testX,2), numTestIms]);
% Store target labels for each attack. The target labels are selected randomly
tgtLabels = zeros(numTestIms, 1);
finalPred_ADAM = zeros(numTestIms,10);
finalPred_NEWTON = zeros(numTestIms,10);

% attempt attack on each test set image
for n = 1:numTestIms
    % extract test image and candidate groundtruth label
    testIm = testX(:,:,:,n);
    groundtruthLabel = testY(n);
    % randomly select target label
    groundtruthLabelIdx = false(1,10);
    groundtruthLabelIdx(groundtruthLabel+1)=true;
    tgt = datasample(classLabels(~groundtruthLabelIdx),1);
    % run the test image attack
    if (strcmp(algorithm, 'ADAM'))
        [isSuccess,numQueries, imFinal, finalPred_ADAM(n,:)] = attack_adam(testIm, tgt, net);
    elseif (strcmp(algorithm, 'Newton'))
        [isSuccess, numQueries, imFinal,finalPred_NEWTON(n,:)] = attack_newton(testIm, tgt,net);
    end

    % store experimental results
    successTrials(n) = isSuccess;
    numQueriesTrials(n) = numQueries;
    imsAdversarial(:,:,n) = imFinal;
    tgtLabels(n) = tgt;
end

%% calculate accuracy and average number of queries for each successful attack
% attack success rate
accuracy = numel(successTrials(successTrials))/numel(successTrials);

% average number of queries for each successful attack
meanNumQueries = mean(numQueriesTrials(successTrials));

% get poisoned images at each successful trial 
imsSuccess = imsAdversarial(:,:,successTrials);

% get target labels at each successful trial
tgtLabelsSuccess = tgtLabels(successTrials);
% save results
save(sprintf('%s_results.mat',algorithm),'accuracy', 'meanNumQueries', 'imsSuccess', 'tgtLabelsSuccess');

%% 
%append adversarial and clean finalPred
% finalPredAll = finalPred;
finalPredNotAd_ADAM = predict(net,testX(:,:,:,numTestIms+1:2*numTestIms));
finalPredAll_ADAM = [finalPred_ADAM; finalPredNotAd_ADAM];

%Append adversarial and clean target labels/ground-truth
tgtLabelsAll = testY(:,1:2*numTestIms);
tgtLabelsAll(1:numTestIms) = tgtLabels(1:numTestIms)';


plotROC(finalPredAll_ADAM,numTestIms,testY,tgtLabelsAll);

%%
%append adversarial and clean finalPred
% finalPredAll = finalPred;
finalPredNotAd_NEWTON = predict(net,testX(:,:,:,numTestIms+1:2*numTestIms));
finalPredAll_NEWTON = [finalPred_NEWTON; finalPredNotAd_NEWTON];

%Append adversarial and clean target labels/ground-truth
tgtLabelsAll = testY(:,1:2*numTestIms);
tgtLabelsAll(1:numTestIms) = tgtLabels(1:numTestIms)';

plotROC(finalPredAll_NEWTON,numTestIms,testY,tgtLabelsAll);

