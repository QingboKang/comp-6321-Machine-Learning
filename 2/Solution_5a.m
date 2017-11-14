% This is the solution for the NO.5 problem part a of Assignment 2
% Using discriminative vs. generative classifiers
% Author: Qingbo Kang 
% Student ID: 40058122
% e-mail: qi_kang@encs.concordia.ca

close all;
clear;

%% a
% load data files
X = load('wpbcx.dat');
y = load('wpbcy.dat');

% add bias term to x
[num_samples, num_features] = size(X);
X = [X, ones(num_samples, 1)];

% initialization
lr = [0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8];
num_iterations = 600;
best_lr = 10;
best_loss = +Inf;
num_folds = 10;

% do CV
indices = crossvalind('Kfold', num_samples, num_folds);

% finding best learning rate
for ii = 1 :length(lr)
    w = normrnd(0, 1, [num_features+1 1]) * 0.001;
    
    for fold = 1 : num_folds
        test_idx = (indices == fold);
        train_idx = ~test_idx;
        
        Xtrain = X(train_idx, :);
        Ytrain = y(train_idx);
        Xtest = X(test_idx, :);
        Ytest = y(test_idx);
        
        % train the logistic regression model, get parameters 
        w = LogisticRegressionTrain(w, Xtrain, Ytrain, lr(ii), num_iterations);
        
        % cost on train and test set
        [Jtrain] = costFunction(w, Xtrain, Ytrain);
        [Jtest] = costFunction(w, Xtest, Ytest);
               
        Jtrain_folds(fold, ii) = Jtrain;
        Jtest_folds(fold, ii) = Jtest;
        
    end
    
    [loss, loss_index] = min(mean(Jtrain_folds));
    
    if(loss < best_loss)
      best_loss = loss;
      best_lr = lr(ii);
      best_lr_index = ii;        
    end
end

% plot the mean errors in training set for different learning rate
plot(lr, mean(Jtrain_folds), 'r.', 'MarkerSize', 20 );
xlabel('Learning rate');
ylabel('Mean error on traing set');

% using the best learning rate, get the errors the full dataset
% during training process
w = normrnd(0, 1, [num_features+1 1]) * 0.001; 
for j = 1:num_iterations
    [J, grad] = costFunction(w, X, y);

    error(j) = J;
    
    w = w - lr(best_lr_index) * grad;
end

% plot
figure;
plot(1:num_iterations, error, 'r.', 'MarkerSize', 10 );
legend('Training error');
title('Training error over iterations for the best learning rate');
xlabel('Iterations');
ylabel('Error');
