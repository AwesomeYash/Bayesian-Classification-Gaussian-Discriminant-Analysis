%% EECE5644 - Assignment 1 
% Name: Priyanshu Ranka (002305396)
% Professor: Deniz Erdogmus
% Problem: Question 3
% Task: 
clear all; close all; clc;

%% MAP Classifier for the RED Wine Quality Dataset

fprintf("-------------------- Wine Data Classification -----------------------------------")

%Load the Dataset
fileName = 'winequality-white.csv';
data = readmatrix(fileName, 'Delimiter', ';', 'NumHeaderLines', 1);

% Getting Features and Labels
x = data(:, 1:11)';      % There are 11 features 
labels = data(:, 12)';   % Labels are in the last column('Quality' of wine)
    
[n, N] = size(x);   % n-Dimensions; N-No of samples
class_labels = unique(labels);
C = length(class_labels);   % No of uniques class labels

% Debugging Step %
fprintf("Datset Loaded! \n");
fprintf("Samples: %d, Features: %d, Unique Classes: %d\n", N,n, C);

% Estimating the Parameters and Regularizing them for all Classes
priors = zeros(1, C);
mu_estimated = zeros(n, C);
Sigma_estimated = zeros(n, n, C);

for i = 1:C
    c = class_labels(i);
    class_indices = find(labels == c);
    
    % Estimating Priors
    priors(i) = length(class_indices) / N;

    % Estimating Mean Vectors and Covariance Matrices
    mu_estimated(:, i) = mean(x(:, class_indices), 2);
    Sigma_estimated(:, :, i) = cov(x(:, class_indices)');

    % Regularization
    lambda = 0.01 * trace(Sigma_estimated(:,:,i))/rank(Sigma_estimated(:,:,i));
    Sigma_estimated(:, :, i) = Sigma_estimated(:, :, i) + lambda * eye(n);
end

% Calculating Likelihoods and Implementing the MAP Classifier
likelihoods = zeros(C, N);
for i = 1:C
    likelihoods(i, :) = evalGaussian(x, mu_estimated(:, i), Sigma_estimated(:, :, i));
end

class_posteriors = likelihoods .* priors';
[~, decisions] = max(class_posteriors, [], 1);
decisions = class_labels(decisions);

% Analysis
confusion_matrix = confusionmat(labels, decisions);
fprintf('Confusion Matrix (Rows: Decisions, Columns: True Labels):\n');
disp(confusion_matrix);

p_error = sum(labels ~= decisions) / N;
fprintf('Overall Probability of Error: %.4f (%.2f%%)\n\n', p_error, p_error*100);

% Visualizing with PCA
muhat_total = mean(x, 2);
xzm = x - muhat_total;
Sigmahat_total = cov(xzm');
[Q, D] = eig(Sigmahat_total);
[d_sorted, ind] = sort(diag(D), 'descend');
Q_sorted = Q(:, ind);
projected_data = Q_sorted' * xzm;

pc1 = projected_data(1, :);
pc2 = projected_data(2, :);
variance_explained = 100 * d_sorted / sum(d_sorted);

figure;
gscatter(pc1, pc2, labels, [], 'o', 6);
title('White Wine Data Projected onto First Two Principal Components');
xlabel(sprintf('Principal Component 1 (%.2f%% variance)', variance_explained(1)));
ylabel(sprintf('Principal Component 2 (%.2f%% variance)', variance_explained(2)));

% Create a dynamic legend for the quality scores
legend_entries = arrayfun(@(c) sprintf('Quality %d', c), class_labels, 'UniformOutput', false);
legend(legend_entries, 'Location', 'northeastoutside');
grid on;
fprintf('PCA plot generated.\n\n');



%% MAP Classifier for the Human Activity Recognition Dataset

fprintf("-------------------- Human Activity Classification -----------------------------------")

% Load training and test data
X_train = readmatrix(fullfile('X_train.txt'));
y_train = readmatrix(fullfile('y_train.txt'));
X_test = readmatrix(fullfile('X_test.txt'));
y_test = readmatrix(fullfile('y_test.txt'));

% Combine training and test sets to use all available data
x = [X_train', X_test']; % Features are transposed to be (dims x samples)
labels = [y_train', y_test']; % Labels

[n, N] = size(x); % n - dimensions; N - No of samples
class_labels = unique(labels);
C = length(class_labels); % Number of unique classes

% Debugging Step %
fprintf("\n Datset Loaded! \n");
fprintf("Samples: %d, Features: %d, Unique Classes: %d\n", N,n, C);

% Estimate and Regularize Parameters for each Class
priors = zeros(1, C);
mu_estimates = zeros(n, C);
Sigma_est = zeros(n, n, C);

for i = 1:C
    c = class_labels(i);
    class_indices = find(labels == c);
    
    % Estimate Priors
    priors(i) = length(class_indices) / N;
    
    % Estimate Mean Vectors amd Covariance Matrices
    mu_estimates(:, i) = mean(x(:, class_indices), 2);
    Sigma_estimates(:, :, i) = cov(x(:, class_indices)');
    
    % Regularization
    lambda = 0.01 * trace(Sigma_estimates(:,:,i))/rank(Sigma_estimates(:,:,i));
    Sigma_estimates(:, :, i) = Sigma_estimates(:, :, i) + lambda * eye(n);
end

% Implement and Apply the MAP Classifier
likelihoods = zeros(C, N);
for i = 1:C
    likelihoods(i, :) = evalGaussian(x, mu_estimates(:, i), Sigma_estimates(:, :, i));
end

class_posteriors = likelihoods .* priors'; 
[~, decisions] = max(class_posteriors, [], 1);
decisions = class_labels(decisions);

% Analysis
confMat = confusionmat(labels, decisions);
fprintf('Confusion Matrix (Rows: Decisions, Columns: True Labels):\n');
disp(confMat);

p_error = sum(labels ~= decisions) / N;
fprintf('Overall Probability of Error: %.4f (%.2f%%)\n\n', p_error, p_error*100);

% Visualize with PCA
muhat_total = mean(x, 2);
xzm = x - muhat_total;
Sigmahat_total = cov(xzm');
[Q, D] = eig(Sigmahat_total);
[d_sorted, ind] = sort(diag(D), 'descend');
Q_sorted = Q(:, ind);
projected_data = Q_sorted' * xzm;

pc1 = projected_data(1, :);
pc2 = projected_data(2, :);
variance_explained = 100 * d_sorted / sum(d_sorted);

figure;
gscatter(pc1, pc2, labels, [], 'o', 6);
title('Human Activity Data Projected onto First Two Principal Components');
xlabel(sprintf('Principal Component 1 (%.2f%% variance)', variance_explained(1)));
ylabel(sprintf('Principal Component 2 (%.2f%% variance)', variance_explained(2)));
legend({'Walking', 'Walking Upstairs', 'Walking Downstairs', 'Sitting', 'Standing', 'Laying'}, 'Location', 'northeastoutside');
grid on;
fprintf('PCA plot generated.\n');