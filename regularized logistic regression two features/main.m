%% Load Data
% The first two columns contain the exam scores and the third column contains the label.
data = load('ex2data2.txt');
X = data(:, [1, 2]); 
y = data(:, 3);

% Plot the data with + indicating (y = 1) examples and o indicating (y = 0) examples.
hold on
admitted = find(y==1);
notAdmitted = find(y==0);
plot(X(admitted,1), X(admitted,2), '+k','markerSize',5);
plot(X(notAdmitted,1), X(notAdmitted,2), 'o','markerSize',5,'MarkerEdgeColor','k','MarkerFaceColor',[0.9 0.9 0]);
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0')

% Add Polynomial Features
% Note that mapFeature also adds a column of ones for us, so the intercept term is handled
X = mapFeature(X(:,1), X(:,2));
%%  Setup the data matrix appropriately
[m, n] = size(X);

% Add intercept term to X
X = [ones(m, 1) X];

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1
lambda = 1;

% Compute and display initial cost and gradient for regularized logistic regression
[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);
fprintf('Cost at initial theta (zeros): %f\n', cost);

%%  Set options for fminunc
% instad of calculating gradient descent manually, here we used internal
% function of MATLAB "fminunc"
% requires Optimization Toolbox
options = optimoptions(@fminunc,'Algorithm','Quasi-Newton','GradObj', 'on', 'MaxIter', 400);
% options = optimoptions('fminunc'); % Start with the default options
%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta);

% Print theta
fprintf('Cost at theta found by fminunc: %f\n', cost);
disp('theta:');disp(theta);

% Plot Boundary
plotDecisionBoundary(theta, X, y)
hold on;
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend('Admitted', 'Not admitted')
hold off;
