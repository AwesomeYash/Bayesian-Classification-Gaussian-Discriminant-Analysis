%% EECE5644 - Assignment 1 
% Name: Priyanshu Ranka (002305396)
% Professor: Deniz Erdogmus
% Problem: Question 2
% Tasks: 1. MAP Classfier  2. ERM Classfication with Custom Loss Matrix

clear all; clc; close all;

%% Task A: Minimum Probability of Error (MAP) Classifier

fprintf("\n-----------------Part A: MAP Classifier-----------------\n");

% Defining the 4 classes
% Reference- ERMwithClabels.m
N = 10000;               % Total number of samples
C = 4;                   % Number of classes
priors = ones(1, C) / C; % Uniform class priors as per assignment

% Defining the mu(Mean) vectors and Sigma(Covariances) (using 2 methods)
% Method 1 - Randomly assigning arbitrary values
% Refernce- Sample code by Professor
mu = 5 * 1.25 * 2 * C *(rand(2,C)); 
for l = 1:C
    A = 5 * eye(2) + randn(2,2);
    Sigma(:,:,l) = A'*A; % arbitrary covariance matrices
end

% Method 2 - Assigning values so that Part B of the question can be challenged
%{
mu(:, 1) = [0, 10];
mu(:, 2) = [-10, -5];
mu(:, 3) = [10, -5];
mu(:, 4) = [0, 0]; 
% For the purpose of assignment I shall be using the method as given in the sample codes by the professor.
% Defining Covariance
Sigma(:, :, 1) = [4, 1; 1, 4];
Sigma(:, :, 2) = [4, -1; -1, 4];
Sigma(:, :, 3) = [4, 1; 1, 4];
Sigma(:, :, 4) = [6, 0; 0, 6];
%}

% Generating Data
% Refernce - ERMwithClabels.m and ExpectedRiskMinimization.m
labels = randsrc(1, N, [1:C; priors]);
x = zeros(2, N);

for i = 1:C
    N1 = sum(labels == i);
    x(:, labels == i) = mvnrnd(mu(:, i), Sigma(:, :, i), N1)';
end

% Calculating the Likelihoods and Posterior Probabilities
% Reference - Shared Computation Logic from ERMwithCLabels.m
likelihood = zeros(C, N);
for i = 1:C
    likelihood(i, :) = evalGaussian(x, mu(:, i), Sigma(:, :, i));
end
px = priors * likelihood;   % Probability Px; According to the formula

% Replace any zeros in px with a small number to avoid division by zero
px(px==0) = 1e-10;
class_posteriors = (likelihood .* priors') ./ px;   % P(L=1|x) formula

% Decisions using MAP Classifier
% Reference - ERMwithCLabels.m
% 0-1 Loss Matrix and Expected Risk
loss_matrix_MAP = ones(C, C) - eye(C);
expected_risk_MAP = loss_matrix_MAP * class_posteriors;

% Finding the best decision -> Min Risk => MAP Rule
[~, decisions_MAP] = min(expected_risk_MAP, [], 1);

% Confusion Matrix
confusion_matrix_MAP = zeros(C, C);
for L = 1:C
    N1 = sum(labels == L);
    for D = 1:C
        Ndl = sum(decisions_MAP == D & labels == L);
        confusion_matrix_MAP(D, L) = Ndl / N1;
    end
end

% Displaying the Confusion Matrix
fprintf('Confusion Matrix (Rows: Decisions, Columns: True Labels):\n');
disp(confusion_matrix_MAP);

% Calculating overall probabiltity of Error from Confusion matrix
p_correct_MAP = sum(diag(confusion_matrix_MAP) .* priors');
p_error_MAP = 1 - p_correct_MAP;
fprintf('Overall P(error) = %.4f\n', p_error_MAP);

% Visualizing the data with classification results
figure;
mShape = '*o^s'; 
correct_decisions = (decisions_MAP == labels);

% Plot for Correctly Classfied Class 
for l = 1:C
    indices = (labels == l & correct_decisions);
    plot(x(1, indices), x(2, indices), mShape(l), "Color", 'g', 'MarkerFaceColor', 'g','DisplayName', sprintf('Correct Class %d', l));
    hold on,
end

% Plot for Incorrectly Classfied Class 
for l = 1:C
    indices = (labels == l & ~correct_decisions);
    plot(x(1, indices), x(2, indices), mShape(l), "Color", 'r', 'MarkerFaceColor', 'r','DisplayName', sprintf('Incorrect Class %d', l));
    hold on,
end

title('Part A: MAP Classification Results (Green=Correct, Red=Incorrect)');
legend('Location', 'northeastoutside');
axis equal; grid on;


%% Part B: ERM Classification with Custom Loss Matrix

fprintf("\n-----------------Part B: ERM Classifier (Custom Loss)-----------------\n");

% Defining the custom loss matrix 
loss_matrix_ERM = [0 1 1 3; 1 0 1 3; 1 1 0 3; 1 1 1 0];
fprintf("Loss Matrix:\n")
disp(loss_matrix_ERM);

% Calculating Expected Risk
expected_risk_ERM = loss_matrix_ERM * class_posteriors;
% Decisions for Minimum Risk
[~, decisions_ERM] = min(expected_risk_ERM, [], 1);

% Confusion Matrix for ERM Classifier
confusion_matrix_ERM = zeros(C, C);
for L = 1:C
    N1 = sum(labels == L);
    for D = 1:C
        Ndl = sum(decisions_ERM == D & labels == L);
        confusion_matrix_ERM(D, L) = Ndl / N1;
    end
end

% Displaying the Confusion Matrix
fprintf('Confusion Matrix (Rows: Decisions, Columns: True Labels):\n');
disp(confusion_matrix_ERM);

% Calculating overall probabiltity of Error from Confusion matrix
p_correct_ERM = sum(diag(confusion_matrix_ERM) .* priors');
p_error_ERM = 1 - p_correct_ERM;
fprintf('Overall P(error) = %.4f\n', p_error_ERM);

% Estimating the Minimum Expected Risk for all 10000 samples
total_loss = 0;
for i = 1:N
    true_label = labels(i);
    decision = decisions_ERM(i);
    total_loss = total_loss + loss_matrix_ERM(decision, true_label);
end

min_expected_risk = total_loss / N;
fprintf('Estimated Minimum Expected Risk (Avg. Loss) = %.4f\n', min_expected_risk);

% Visualizing the data with ERM classification results
figure;
correct_decisions_ERM = (decisions_ERM == labels);

% Plot for Correctly Classfied Class 
for l = 1:C
    indices = (labels == l & correct_decisions_ERM);
    plot(x(1, indices), x(2, indices), mShape(l), "Color", 'g', 'MarkerFaceColor', 'g','DisplayName', sprintf('Correct Class %d', l));
    hold on,
end

% Plot for Incorrectly Classfied Class 
for l = 1:C
    indices = (labels == l & ~correct_decisions_ERM);
    plot(x(1, indices), x(2, indices), mShape(l), "Color", 'r', 'MarkerFaceColor', 'r','DisplayName', sprintf('Incorrect Class %d', l));
    hold on,
end
title('Part B: ERM Classification Results (Green=Correct, Red=Incorrect)');
legend('Location', 'northeastoutside');
axis equal; grid on;

%% Helping Functions
function g = evalGaussian(x, mu, Sigma)
    % Evaluates the Gaussian pdf N(mu,Sigma) at each column of x
    [n, ~] = size(x);
    C = ((2*pi)^n * det(Sigma))^(-1/2);
    E = -0.5 * sum((x - mu) .* (inv(Sigma) * (x - mu)), 1);
    g = C * exp(E);
end
