% Implement k-nearest neighbor on the Wisconsin data set, using Euclidean 
% distance. Perform cross-validation for different values of k and plot 
% training and testing error curves as a function of k.

clc; clear;
close all;

x = load('./wpbcx.dat');
y = load('./wpbcy.dat');

[num_samples, num_features] = size(x);
num_folds = 10;
k_nearest = 30;

indices = crossvalind('Kfold', num_samples, num_folds);
k_errors = zeros(k_nearest, 2);
for k = 1:k_nearest
    train_error = zeros(num_folds, 1);
    test_error = zeros(num_folds, 1);
    for ii = 1:num_folds
        test_idx = (indices == ii);
        train_idx = ~test_idx;

        Xtrain = x(train_idx, :);
        Ytrain = y(train_idx);
        Xtest = x(test_idx, :);
        Ytest = y(test_idx);

        pred_train = knn(Xtrain, Ytrain, k, Xtrain);
        pred_test = knn(Xtrain, Ytrain, k, Xtest);
       
        train_error(ii) = mean(pred_train ~= Ytrain);
        test_error(ii) = mean(pred_test ~= Ytest);
    end
    k_errors(k, 1) = mean(train_error);
    k_errors(k, 2) = mean(test_error);
end

plot(1:k_nearest, k_errors(:, 1), 1:k_nearest, k_errors(:, 2));
title('10-fold cross-validation on k');
xlabel('k');
ylabel('Cross-validation Errors');
legend('Training Error', 'Testing Error');