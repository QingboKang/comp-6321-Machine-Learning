function [ theta ] = LogisticRegressionTrain( theta, X, y, lr, ...
    num_iterations )
%LOGISTICREGRESSIONTRAIN 

for j = 1 : num_iterations
    [J, grad] = costFunction(theta, X, y);
    theta = theta - lr * grad;
end


end

