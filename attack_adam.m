function [isSuccess, numQueries, imFinal,finalPred] = attack_adam(im, tgt, network)
% Attack a test image and misclassify it to the target class.
% Inputs:
% im: MxN grayscale image to attack.
% tgt: index of target class. The adversarial attack attempts to make
%   the classifer assign the target class to im. network: trained neural network image classifier
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
BATCH_SIZE = 128;

% exponential decay rates for moment estimates
beta1 = 0.9;
beta2 = 0.999;
epsilon = 1e-8;
%% run ADAM iterations
% initialize ADAM states for each pixel in input image
M = zeros(size(im));
v = zeros(size(im));
T = zeros(size(im));

% change image intensities to be in range [0,1]
im = double(im./255);

% initial loss
lossOld = inf;
lossNew = evaluateLoss(im, tgt, KAPPA, network);

% count the number of times the model is queried
numQueries = 0;
% indicate whether or not the attack is successful
isSuccess = false;
counter = 0;
% add noise to image until convergence 
while((abs(lossOld - lossNew) >= CONVERGENCE_THRESH)||counter == 9)
    % Get random batch of pixel coordinates to update. Note that here pixel
    % coordinates are represented as linear indices, not row/column subscripts
    pixelBatch = datasample(1:numel(im), BATCH_SIZE, 'Replace', false);

    % calculate gradient at current pixel batch 
    [grad,~] = evaluateLossGradient(im, pixelBatch, network, KAPPA, tgt) ;
    % update ADAM states for pixels in current batch
    T(pixelBatch) = T(pixelBatch) + 1;
    M(pixelBatch) = beta1*M(pixelBatch)+(1-beta1)*grad;
    v(pixelBatch) = beta2*v(pixelBatch)+(1-beta2)*(grad.^2);
    
    % bias corrected first and second moment estimates
    M_hat = M(pixelBatch)./(1-beta1.^(T(pixelBatch)));
    v_hat = v(pixelBatch)./(1-beta2.^(T(pixelBatch)));
    % take a gradient descent step for current pixel batch
    im(pixelBatch) = im(pixelBatch) - STEP_SIZE*M_hat./(sqrt(v_hat)+epsilon);
%     imshow(im)
    counter = counter+1;
    % clip image values to be in range [0,1]
    im(im < 0) = 0;
    im(im > 1) = 1;
    
    % update the count of model queries
    numQueries = numQueries + 2*BATCH_SIZE;

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
[~, finalPrediction] = max(finalPrediction);

% attack is successful when the model score for the target class is higher than all other class scores 
if (finalPrediction == tgt)
    isSuccess = true;
end

end

