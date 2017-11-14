function [ J, grad ] = costFunction( theta, X, y )
%COSTFUNCTION Compute cost and gradient for logistic regression 

% Initialize some useful values
m = length(y); % number of training examples


J = 0;
grad = zeros(size(theta));

predictions = sigmoid( X * theta );


J = (-1 / m) * sum( y' * log(predictions) + (1 - y)' * log(1 - predictions) );

for ii = 1 : size(theta)
    grad(ii) = 1/m * (predictions - y)' * X(:, ii);
end

end

