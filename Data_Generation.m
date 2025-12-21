%% EECE5644 - Assignment 1 - Data Generation Script

clear all, close all,

N = 10000;
p0 = 0.65;
p1 = 0.35;
labels = rand(1,N)>=p0; 
N0 = length(find(labels==0)); 
N1 = length(find(labels==1));

mu(:,1) = [-1/2; -1/2; -1/2]; 
Sigma(:,:,1) = [1,-0.5,0.3; -0.5,1,-0.5; 0.3,-0.5,1];
r0 = mvnrnd(mu(:,1), Sigma(:,:,1), N0);
figure(1),plot3(r0(:,1),r0(:,2),r0(:,3),'.b');axis equal,hold on,

mu(:,2) = [1; 1; 1];
Sigma(:,:,2) = [1,0.3,-0.2; 0.3,1,0.3; -0.2,0.3,1];
r1 = mvnrnd(mu(:,2), Sigma(:,:,2), N1);
figure(1),plot3(r1(:,1),r1(:,2),r1(:,3),'.r');axis equal,hold on,

% Combinig into a single matrix
x = zeros(3, N);
x(:,labels == 0) = r0';
x(:,labels == 1) = r1';

% Saving data 
save('data.mat', 'x', 'labels', 'p0', 'N0', 'N1', 'mu', 'Sigma', 'r0', 'r1');
