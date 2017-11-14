function [ prob ] = GaussianPDF( X, mu, sigma )
%GAUSSIANPDF pdf of Gaussian distribution
% 

exp_term = exp( (-(X - mu)^2) / (2 * sigma^2) );

prob = (1 / (sigma * sqrt(2*pi))) * exp_term;

end

