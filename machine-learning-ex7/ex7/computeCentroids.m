function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returs the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);
centroids = [];


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points nthat
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%


for i=1:K
  class_idx = find(idx==i);
  class_X = X(class_idx,:);
  class_size = size(class_idx,1);
  class_centroid = sum(class_X,1) * (1/class_size);
  centroids = [centroids;class_centroid];
end

end

