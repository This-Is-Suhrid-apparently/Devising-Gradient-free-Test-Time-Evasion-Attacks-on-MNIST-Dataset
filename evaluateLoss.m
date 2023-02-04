function loss = evaluateLoss (x, tgt, kappa, network)
% Evaluate loss function for a single data sample/target.
% Inputs:
% x: image to classify, scaled to range [0,1].
% tgt: index of target class. The adversarial attack attempts to make
% the classifer assign the target class to x.
% kappa: Scalar threshold parameter, must be >= 0 network: trained neural network image classifier
% Outputs:
% loss: Scalar, loss function evaluated at input parameters
% Get model outputs on data sample x. The image intensities are rescaled to % the range [0,255] because the network is trained on images of that scale.

modelOut = predict (network, x.*255);

% get index for target class score in model output
tgtIdx = false(size(modelOut));
tgtIdx(tgt) = true;

tgtScore = modelOut(tgtIdx);
nonTgtScores = modelOut(~tgtIdx);

maxTgtScoreDiff = max(log(nonTgtScores)-log(tgtScore));
loss = max(maxTgtScoreDiff, -kappa);

end

