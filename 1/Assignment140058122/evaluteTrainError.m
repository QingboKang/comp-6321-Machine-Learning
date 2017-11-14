function [ J ] = evaluteTrainError( X, y, weight )
% EVALUTETRAINERROR Evalute the training error of the resulting fit
%     J = evaluteTrainError( X, y, weight ) computes the cost of using 
%     weight as the parameter for linear regression to fit the data points
%     in X and y
% 

% make sure the X and weight have the same size
targetSize = size(weight, 1);
XSize = size(X, 2);

if XSize ~= targetSize
   for ii = 2: 1 + (targetSize - XSize) 
       X = [ X(:, end - 1).^ii, X];
   end
end

% compute predictions
predictions = X * weight;

% obtain the training error 
J = (1/2) * (predictions - y)'*(predictions - y);


end

