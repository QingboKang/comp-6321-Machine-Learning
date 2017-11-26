% K-means
% Question 3 of Assignment 4 of COMP 6321 Machine Learning

clc, clear;
close all;

% read data 
data = load('hw4-image.txt');

K = 8;
centroids = [255, 255, 255; 
    255, 0, 0;
    128, 0, 0;
    0, 255, 0;
    0, 128, 0;
    0, 0, 255;
    0, 0, 128;
    0, 0, 0];

[idx, new_centroids] = Kmeans(data, K, centroids);

new_img = new_centroids(idx, :);


num = zeros(size(new_centroids, 1), 1);
for ii = 1 : size(new_centroids, 1)
    num(ii) = size( find(idx == ii), 1 );
end

% how many clusters there are in the end
cluster_num = sum( num~= 0 );
fprintf('\n\nThere are %d clusters in the end.\n\n', cluster_num);

% the final centroids of each cluster
fprintf('The final centroids of each cluster:\n');
disp(new_centroids);

% the number of pixels associated to each cluster
fprintf('\nThe number of pixels associated to each cluster:\n');
disp(num);

% display image
new_img = uint8(new_img);
new_img = reshape(new_img, 407, 516, 3);
image(flip(imrotate(new_img, -90), 2));
title('Resulting image');


