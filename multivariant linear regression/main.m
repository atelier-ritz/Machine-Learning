%% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y); % number of samples

% [optional] Plot some data points
% figure;
% plot(X(1:10,1),y(1:10,:),'rx','MarkerSize',10);
% figure;
% plot(X(1:10,2),y(1:10,:),'rx','MarkerSize',10);

%% Feature Normalization
% Scale features and set them to zero mean and one standard deviation
[X, mu, sigma] = featureNormalize(X);
X = [ones(m, 1) X]; % Add intercept term to X

%% Grandient Descent
% Run gradient descent
alpha = 0.1;% learing rate
num_iters = 400;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
fprintf('Theta computed from gradient descent:\n%f,\n%f \n',theta(1),theta(2))

% [optional] plot tendency of cost function
% plot(1:num_iters, J_history, '-r', 'LineWidth', 2)
% xlabel('Number of iterations');
% ylabel('Cost Function J');

%% Prediction
% Estimate the price of a 1650 sq-ft, 3 br houseinput = [1650,3];
input = [1650,3];
input_norm = (input - mu)./sigma; % Normalize input
input_norm = [1, input_norm]; % Add intercept term to X
price = input_norm * theta; 
fprintf('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n $%f \n', price);