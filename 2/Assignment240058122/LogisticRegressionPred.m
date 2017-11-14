function [ pred ] = LogisticRegressionPred( w, X )
%LOGISTICREGRESSIONPRED Predict the ouput Y based on the weight using
%logistic regression
%   

% calculation predictions
pred = sigmoid(X*w);
pred = (pred > 0.5);

end

