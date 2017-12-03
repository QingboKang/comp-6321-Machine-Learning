clc, clear;
close all;

%% Playing with PCA
% a
mu = [5, 20];
sigma = [10, 2; 2, 5];
data = mvnrnd(mu, sigma, 200);

[COEFF_A, SCORE, latent, tsquare] = princomp(data);

plot(data(:, 1), data(:, 2), 'r.', 'MarkerSize', 10);

% b 
% substract the mean from all the data points 
sub_result = bsxfun(@minus, data, mu);
% compute PCA
[COEFF_B, SCORE, latent, tsquare] = princomp(sub_result);

figure;
hold on;
plot(COEFF_B);
scatter(sub_result(:, 1), sub_result(:,2) );
hold off;

% c
data_C(:, 1) = sub_result(:, 1) ./ std(data(:, 1));
data_C(:, 2) = sub_result(:, 2) ./ std(data(:, 2));
% compute PCA
[COEFF_C, SCORE_C, latent_C, tsquare_C] = princomp(data_C);

figure;
hold on;
plot(COEFF_C);
scatter(data_C(:, 1), data_C(:,2) );
hold off;

% d
data_D(:, 2) = data_C(:, 2) * 1000;
data_D(:, 1) = data_C(:, 1);
% Compute PCA
[COEFF_D, SCORE_D,latent_D, tsquare_D] = princomp(data_D); 

figure;
hold on;
plot(COEFF_D);
scatter(data_D(:, 1), data_D(:, 2));
hold off;