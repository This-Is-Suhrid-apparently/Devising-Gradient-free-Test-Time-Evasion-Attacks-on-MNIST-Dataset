function [isSuccess, numQueries, imFinal,finalPred] = attack_newton (im,tgt,network)
% Attack a test image and misclassify it to the target class.
% Inputs:
% im: MxN grayscale image to attack.
% tgt: index of target class. The adversarial attack attempts to make
% the classifer assign the target class to im. network: trained neural network image classifier
% Outputs:
%
% isSuccess: Scalar boolean, set to true if the attack is successful
% and false otherwise.
% numQueries: Scalar integer, containing the number of network queries
% required for the attack.
% imFinal: MN grayscale image with adversarial noise.

%% hyperparameters
KAPPA = 0;
% STEP SIZE = 1e-2;
STEP_SIZE = 0.01;
CONVERGENCE_THRESH = 1e-6;
% BATCH SIZE = 784;
BATCH_SIZE = 128;

%% run newton iterations

% change image intensities to be in range [0,1]
im = double(im)./255;

% initial loss
lossOld = inf;
lossNew = evaluateLoss(im, tgt, KAPPA, network);

% count the number of times the model is queried
numQueries = 0;
% indicate whether or not the attack is successful 
isSuccess = false;

% add noise to image until convergence 
while(abs(lossOld - lossNew) >= CONVERGENCE_THRESH)
    % get random batch of pixel coordinates to update. Note that here 
    % coordinates are represented as linear indices, not now/column subscripts 
    pixelBatch = datasample(1:numel(im), BATCH_SIZE, 'Replace',false);

    % calculate gradient and hessian at each pixel in current batch
    [grad, hessian] = evaluateLossGradient(im,pixelBatch, network, KAPPA,tgt, 'returnHessian', true);

    % don't use negative hessian values 
    hessian(hessian <= 0) = 1;
    % don't allow hessian to be too small due to numerical issues 
    hessian(hessian < 1e-2) = 1e-2;
    % take a gradient descent step for current pixel batch
    im(pixelBatch) = im(pixelBatch) - STEP_SIZE*(grad./hessian);
    % clip image values to be in range [0,1] 
    im(im < 0) = 0;
    im(im > 1) = 1;

    % Update the count of model queries. The gradient calculations require 
    % 2 queries per pixel, and the hessian calculation requires 1 
    % additional query
    numQueries = numQueries + 2*BATCH_SIZE + 1;
    
    % evaluate loss at current iteration
    lossOld = lossNew;
    lossNew = evaluateLoss(im, tgt, KAPPA, network);
end
% Image with added adversarial noise. Also rescale image to have the 
% original range of values (0 to 255).
imFinal = im.*255;

% get predicted class for modified image 
finalPrediction = predict(network, imFinal); 
finalPred = finalPrediction;
[~,finalPrediction] = max(finalPrediction);

% attack is successful when the model score for the target class is higher than all other class score 
if (finalPrediction == tgt)
    isSuccess = true;
end

end

