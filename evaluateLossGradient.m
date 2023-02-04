function [grad, hessian] = evaluateLossGradient (im,batch,network, kappa, tgt,varargin)
% Evaluate gradient of loss function for current image.
% Inputs:
% im: image to classify, scaled to range [0,1].
% batch: Length N linear indices of pixel coordinates. These are the
%
%
% pixels that are updated at current iteration.
% network: trained neural network image classifier kappa: Scalar threshold parameter, must be >= 0
% tgt: index of target class. The adversarial attack attempts to make
% the classifer assign the target class to im.
% vanargin - (Optional) Name, value pairs:
% returnHessian" scalar boolean - set true to return second
% derivatives wrt each coordinate, set false otherwise.
% The default is false.
%
%
%
%
% Outputs:
% grad: Length N vector of loss function gradients wrt to each pixel in
% current batch.
% hessian: If 'returnHessian is true, this is a length N vector of
% second derivatives wrt each coordinate in batch. Otherwise, this is NaN. Note that this is not the full hessian matrix, just the diagonal.

%% Parse varargin
arg_in = inputParser;
addParameter (arg_in,'returnHessian', false, @(x)(islogical(x) && numel(x) == 1));
parse(arg_in, varargin{:});

returnHessian = arg_in.Results.returnHessian;

%% constant step parameter for calculating finite differences
h = 1e-6;%1e-5;

%% calculate gradients
grad = zeros(size(batch));
% store f(x-h*e_i) for each pixel in batch (used for hessian calculation)
forwardsLoss = zeros(size(batch));
% store f(x-h*e_i) for each pixel in batch (used for hessian calculation)
backwardsLoss = zeros(size(batch));
% calculate gradient wt each pixel coordinate in batch
for n = 1: length(batch)
    forwardsIm = im;
    forwardsIm(batch(n)) = forwardsIm(batch(n))+h;
    backwardsIm = im;
    backwardsIm(batch(n)) = backwardsIm(batch(n))-h;
    % f(x+h*e_n)
    forwardsLossCurr = evaluateLoss(forwardsIm, tgt, kappa,network);
    % f(x-h*e_n)
    backwardsLossCurr = evaluateLoss(backwardsIm, tgt, kappa, network);
    % store gradients and network queries
    grad(n) = (forwardsLossCurr - backwardsLossCurr)/(2*h);
    forwardsLoss(n) = forwardsLossCurr;
    backwardsLoss(n) = backwardsLossCurr;
end

%% calculate hessians

if (returnHessian)
    % loss evaluated on unmodified image
    loss = evaluateLoss(im, tgt, kappa, network);
    % second derivatives wrt each coordinate in batch
    hessian = (forwardsLoss + backwardsLoss - 2*loss)/h^2;
else
    hessian = nan;
end

end
