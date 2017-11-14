function [ prior_true, prior_false, m_true, m_false, std_true, std_false...
    ] = GaussianNaiveBayesTrain( X, y )
%GAUSSIANNAIVEBAYESTRAIN Training the Gaussian Naive Bayes 
% (binary classification)

% the probabilities of class true & false (prior)
prior_true = mean(y==1);
prior_false = mean(y==0);

% mean values for a given class value
m_true = mean( X(y == 1, :) );
m_false = mean( X(y == 0, :) );

% standard deviation for a given class value
std_true = std( X(y == 1, : ) );
std_false = std( X(y == 0, :) );

end

