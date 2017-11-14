function [ pred ] = GaussianNaiveBayesPredict( X, prior_true, ...
    prior_false, m_true, std_true, m_false, std_false )
%GAUSSIANNAIVEBAYESPREDICT 
%   
[num_samples, num_features] = size(X);

% final predictions
pred = zeros(num_samples, 1);

for ii = 1 : num_samples
    prob_true = 1;
    prob_false = 1;
    for jj = 1: num_features
        prob_true = prob_true * GaussianPDF( X(ii, jj), ...
            m_true(jj), std_true(jj) );
        prob_false = prob_false * GaussianPDF( X(ii, jj), ...
            m_false(jj), std_false(jj) );
    end
    % class probabilities
    prob_post_true = (prob_true * prior_true) / ( prob_true * prior_true + ...
        prob_false * prior_false );
    prob_post_false = (prob_false * prior_false) / ( prob_true * prior_true + ...
        prob_false * prior_false );
    
    % prediction
    if (prob_post_true > prob_post_false)
        pred(ii) = 1;
    else
        pred(ii) = 0;
    end
end

end

