function [ cluster_idx, centroids ] = Kmeans( data, K, centroids )
%KMEANS

[num_samples, ~] = size(data);
cluster_idx = zeros(num_samples, 1);
    
iterations = 100;
epsilon = 0.01;
initial_centroids = centroids;
dist = zeros(K, 1);

for it = 1:iterations
    % assign each sample to the nearest cluster
    for i = 1:num_samples
        for j = 1:K
            dist(j) = norm(data(i, :) - centroids(j, :))^2;
        end
        [~, cluster_idx(i)] = min(dist);
    end
    
    % update centroids
    ssd = 0;
    for i=1:K
        indices = find(cluster_idx == i);
        if isempty(indices)
            new_centroids(i, :) = initial_centroids(i, :);
        else
            new_centroids(i, :) = mean(data(indices, :));
            ssd = ssd + norm(bsxfun(@minus, new_centroids(i, :), data(indices, :)))^2;
            %disp(ssd);
        end
    end
    fprintf( 'Iteration: %d\t Sum of squared distance: %f\n', it, ssd);
   % disp(norm(new_centroids - centroids));
    if norm(new_centroids - centroids) <= epsilon
       break; 
    end
    centroids = new_centroids;
end

end

