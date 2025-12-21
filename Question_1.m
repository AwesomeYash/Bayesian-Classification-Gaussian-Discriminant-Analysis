%% EECE5644 - Assignment 1 
% Name: Priyanshu Ranka (002305396)
% Professor: Deniz Erdogmus
% Problem: Question 1
% Tasks: 1. ERM Classfier  2. Naive Bayes Classfier  3. LDA Classfier

clear all, close all, clc;

% Data Loading
load('data.mat');

% Data Visualization
figure(1), plot3(x(1,labels == 0), x(2,labels == 0), x(3,labels == 0),'.b', 'DisplayName', 'Class 0');
hold on;
figure(1), plot3(x(1,labels == 1), x(2,labels == 1), x(3,labels == 1),'.r', 'DisplayName', 'Class 1');
axis equal; grid on; hold on;
title("3D Scatter Plot of the Generated Data");
xlabel('X1'); ylabel('X2'); zlabel('X3');
legend;

%% Part A:  ERM classification using the knowledge of true data pdf

fprintf("\n-----------------Part A: ERM Classifier (True Knowledge)-----------------\n");

% Discriminant score for ERM
% Formula -> discriminant_score = likelihood1/likelihood0;
discriminant_score = log((evalGaussian(x, mu(:,2), Sigma(:,:,2)))) - log((evalGaussian(x, mu(:,1), Sigma(:,:,1)))); % - log(gamma);

% Gamma, TPR and FPR calculation
gamma = logspace(-10, 10, 1001);
tpr_ERM = zeros(1, length(gamma));
fpr_ERM = zeros(1, length(gamma));
p_error_ERM = zeros(1, length(gamma));

for i = 1:length(gamma)
    decisions = discriminant_score >= log(gamma(i));   
    ind00 = find(decisions == 0 & labels == 0); p00 = length(ind00)/N0; % Probability of True Negative
    ind10 = find(decisions == 1 & labels == 0); p10 = length(ind10)/N0; % Probability of False Positive
    ind01 = find(decisions == 0 & labels == 1); p01 = length(ind01)/N1; % Probability of False Negative
    ind11 = find(decisions == 1 & labels == 1); p11 = length(ind11)/N1; % Probability of True Positive
    
    % Saving values of P(error), TPR and FPR in arrays
    fpr_ERM(i) = p10;
    tpr_ERM(i) = p11;
    p_error_ERM(i) = p10 * p0 + p01 * (1 - p0);
end

% Finding the optimal solutions from the values obtained
[min_p_error, ind_ERM] = min(p_error_ERM);
min_error_tpr = tpr_ERM(ind_ERM);
min_error_fpr = fpr_ERM(ind_ERM);
optimal_gamma = gamma(ind_ERM);
theoritical_optimal_gamma = p0/(1 - p0);

% Displaying the Optimal Solutions
fprintf('RESULTS :\n')
fprintf('ERM Minimum P(error): %.4f\n', min_p_error);
fprintf('Empirical Optimal Gamma: %.4f\n', optimal_gamma);
fprintf('Theoritical Optimal Gamma: %.4f\n', theoritical_optimal_gamma);


%% Part B:  Naive Bayesian Classifier 
% ERM classification using incorrect knowledge of data distribution 

fprintf("\n-----------------Part B: ERM Classifier (Incorrect Knowledge)-----------------\n");

% Initializing the Sigma matrix
Sigma_NB(:, :, 1) = eye(3);
Sigma_NB(:, :, 2) = eye(3);

% Calculating the Discriminant Scores
discriminant_score_NB = log((evalGaussian(x, mu(:,2), Sigma_NB(:,:,2)))) - log((evalGaussian(x, mu(:,1), Sigma_NB(:,:,1)))); % - log(gamma);

% TPR and FPR calculation
gamma = logspace(-10, 10, 1001);
tpr_NB = zeros(1, length(gamma));
fpr_NB = zeros(1, length(gamma));

for i = 1:length(gamma)
    decisions = discriminant_score_NB >= log(gamma(i));
    ind00 = find(decisions == 0 & labels == 0); p00 = length(ind00) / N0; % Probability of True Negative
    ind10 = find(decisions == 1 & labels == 0); p10 = length(ind10) / N0; % Probability of False Positive
    ind01 = find(decisions == 0 & labels == 1); p01 = length(ind01) / N1; % Probability of False Negative
    ind11 = find(decisions == 1 & labels == 1); p11 = length(ind11) / N1; % Probability of True Positive
    
    % Saving values of P(error), TPR and FPR in arrays
    fpr_NB(i) = p10;
    tpr_NB(i) = p11;
    p_error_NB(i) = p10 * p0 + p01 * (1 - p0);
end

% Finding the optimal solutions from the values obtained
[min_p_error_NB, ind_NB] = min(p_error_NB);
min_error_tpr_NB = tpr_NB(ind_NB);
min_error_fpr_NB = fpr_NB(ind_NB);
optimal_gamma = gamma(ind_NB);

% Displaying the Optimal Solutions
fprintf('\nRESULTS :\n')
fprintf('Naive Bayes Minimum P(error): %.4f\n', min_p_error_NB);
fprintf('Optimal Gamma: %.4f\n', optimal_gamma);


%% Part C:  Fisher LDA Classifier

fprintf("\n-----------------Part C: Fisher LDA-----------------\n");

% Initializing the mu and Sigma values
mu_LDA(:, 1) = mean(x(:, labels == 0), 2);
mu_LDA(:, 2) = mean(x(:, labels == 1), 2);
Sigma_LDA(:, :, 1) = cov(x(:, labels == 0)');
Sigma_LDA(:, :, 2) = cov(x(:, labels == 1)');

% Between-class and Within-class Scatter Matrix
Sb = (mu_LDA(:,1) - mu_LDA(:,2)) * (mu_LDA(:,1) - mu_LDA(:,2))';
Sw = Sigma_LDA(:,:,1) + Sigma_LDA(:,:,1);

% Fisher LDA projection to get discriminant scores
[V, D] = eig(inv(Sw) * Sb);
[~, max_eig_ind] = max(diag(D));
w_LDA = V(:, max_eig_ind);

discriminant_score_LDA = w_LDA' * x;
if mean(discriminant_score_LDA(labels == 1)) < mean(discriminant_score_LDA(labels == 0))
    w_LDA = -w_LDA;
    discriminant_score_LDA = -discriminant_score_LDA;
end

% Threshold values and Calculating TPR, FPR and P(error) using the ROC Curve
tau_values = linspace(min(discriminant_score_LDA), max(discriminant_score_LDA), 1001);
[tpr_LDA, fpr_LDA, p_error_LDA] = roc_curve(discriminant_score_LDA, labels, tau_values, p0, N0, N1);

[min_p_error_LDA, ind_LDA] = min(p_error_LDA);
min_error_fpr_LDA = fpr_LDA(ind_LDA);
min_error_tpr_LDA = tpr_LDA(ind_LDA);

% Displaying the Optimal Solutions
fprintf('\nRESULTS :\n')
fprintf('LDA Minimum P(error): %.4f\n', min_p_error_LDA);

% Visualizing and Comparing the Minimum P(Error) by all the classifiers
figure;
plot(fpr_ERM, tpr_ERM, 'b-', 'LineWidth', 2, 'DisplayName', 'A: ERM (True PDF)');
hold on;
plot(fpr_NB, tpr_NB, 'g--', 'LineWidth', 2, 'DisplayName', 'B: Naive Bayes');
plot(fpr_LDA, fpr_LDA, 'm:', 'LineWidth', 2, 'DisplayName', 'C: Fisher LDA');
plot(min_error_fpr, min_error_tpr, 'r*', 'MarkerSize', 12, 'LineWidth', 2, 'DisplayName', 'Min P(error) ERM');
plot(min_error_fpr_NB, min_error_tpr_NB, 'ro', 'MarkerSize', 12, 'LineWidth', 2, 'DisplayName', 'Min P(error) NB');
plot(min_error_fpr_LDA, min_error_tpr_LDA, 'rs', 'MarkerSize', 12, 'LineWidth', 2, 'DisplayName', 'Min P(error) LDA');
xlabel('False Positive Rate (FPR)');
ylabel('True Positive Rate (TPR)');
title('ROC Curves for All Classifiers');
legend('Location', 'southeast');
grid on; axis equal; axis([0 1 0 1]);


% Plot Probability of Error vs. Threshold for ERM
figure;
semilogx(gamma, p_error_ERM, 'b-', 'LineWidth', 2, 'DisplayName', 'ERM Error');
hold on;
semilogx(gamma, p_error_NB, 'g--', 'LineWidth', 2, 'DisplayName', 'Naive Bayes Error');

plot([optimal_gamma, optimal_gamma], [0,1], 'r--', 'DisplayName', 'Min Error Threshold (ERM)');
xlabel('Threshold (\gamma)');
ylabel('Probability of Error');
title('P(error) vs. Threshold');
legend('Location', 'north');
grid on;
ylim([0, 0.5]);





%% Helping Functions
function g = evalGaussian(x, mu, Sigma)
    % Evaluates the Gaussian pdf N(mu,Sigma) at each column of x
    [n, ~] = size(x);
    C = ((2*pi)^n * det(Sigma))^(-1/2);
    E = -0.5 * sum((x - mu) .* (inv(Sigma) * (x - mu)), 1);
    g = C * exp(E);
end

function [tpr, fpr, p_error] = roc_curve(scores, labels, thresholds, p0, N0, N1)
    % Initializes arrays to store results
    tpr = zeros(1, length(thresholds));
    fpr = zeros(1, length(thresholds));
    p_error = zeros(1, length(thresholds));

    % Loop through each threshold to calculate performance metrics
    for i = 1:length(thresholds)
        decisions = scores >= thresholds(i);
        
        ind10 = find(decisions==1 & labels==0); p10 = length(ind10)/N0; % False Positive Rate
        ind01 = find(decisions==0 & labels==1); p01 = length(ind01)/N1; % False Negative Rate
        ind11 = find(decisions==1 & labels==1); p11 = length(ind11)/N1; % True Positive Rate
        
        fpr(i) = p10;
        tpr(i) = p11;
        p_error(i) = p10 * p0 + p01 * (1 - p0);
    end
end