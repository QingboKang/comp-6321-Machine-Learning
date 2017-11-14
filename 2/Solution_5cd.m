% This is the solution for the NO.5 problem part c.d of Assignment 2
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

[num_samples, num_features] = size(X);

% use all features
X = [X, ones(num_samples, 1)];

lr = 1.4;
num_iterations = 4000;

% 10-fold cross-validation
num_folds = 10;
indices = crossvalind('Kfold', num_samples, num_folds);

accuracy = zeros(num_folds, 4);

for fold = 1 : num_folds
    
    w = normrnd(0, 1, [num_features+1 1]) * 0.001;
    
    test_idx = (indices == fold);
    train_idx = ~test_idx;
        
    Xtrain = X(train_idx, :);
    Ytrain = y(train_idx);
    Xtest = X(test_idx, :);
    Ytest = y(test_idx);
    
    % logistic regression 
    w = LogisticRegressionTrain(w, Xtrain, Ytrain, lr, num_iterations);
    logistic_reg_pred_train = LogisticRegressionPred(w, Xtrain);
    logistic_reg_pred_test = LogisticRegressionPred(w, Xtest);
    
    logistic_train_accuracy = mean(logistic_reg_pred_train == Ytrain);
    logistic_test_accuracy = mean(logistic_reg_pred_test == Ytest);
    
    % Gaussian Naive Bayes
    [ prior_true, prior_false, m_true, m_false, std_true, std_false] ...
        = GaussianNaiveBayesTrain( Xtrain, Ytrain );
    gau_naive_bayes_pred_train = GaussianNaiveBayesPredict(Xtrain, ...
        prior_true, prior_false, m_true, std_true, m_false, std_false);
    gau_naive_bayes_pred_test = GaussianNaiveBayesPredict(Xtest, ...
        prior_true, prior_false, m_true, std_true, m_false, std_false);
    
    gnb_train_accuracy = mean(gau_naive_bayes_pred_train == Ytrain);
    gnb_test_accuracy = mean(gau_naive_bayes_pred_test == Ytest);
    
    accuracy(fold, :) = [logistic_train_accuracy, logistic_test_accuracy, ...
        gnb_train_accuracy, gnb_test_accuracy];
   
    
end

disp(accuracy);
m_accu = mean(accuracy);
fprintf( 'Logistic regression prediction accuracy on training set(average): %f\n', m_accu(1) );
fprintf( 'Logistic regression prediction accuracy on test set(average): %f\n', m_accu(2) );
fprintf( 'Gaussian Naive Bayes prediction accuracy on training set(average): %f\n', m_accu(3) );
fprintf( 'Gaussian Naive Bayes prediction accuracy on test set(average): %f\n', m_accu(4) );