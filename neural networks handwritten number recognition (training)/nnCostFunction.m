function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% forward propogation
Z2 = [ones(m,1) X] * Theta1'; % m by 25
A2 = sigmoid(Z2);% m by 25
Z3 = [ones(m,1) A2] * Theta2'; % m by 10
A3 = sigmoid(Z3);% m by 10

% convert output
y_mod = zeros(m, num_labels);
for i = 1:m
   y_mod(i,y(i)) = 1; 
end

% computing cost function
J1 = 1 / m * (-y_mod .* log(A3) - (1-y_mod) .* log(1-A3)); % m by 10, without regularization terms
J1 = sum(J1(:));

fMatSqrSum = @(X) sum(sum(X.^2)); % return the sum of all elements squared in a matrix
J2 = lambda / (2 * m) * (fMatSqrSum(Theta1(:,2:end)) + fMatSqrSum(Theta2(:,2:end)));  % regularization terms

J = J1 + J2;

% back propogation
DELTA_1 = zeros(size(Theta1)); % 25 by 401, derivative of cost function with respect to Theta1
DELTA_2 = zeros(size(Theta2)); % 10 by 26, derivative of cost function with respect to Theta2 
for i = 1:m
    % forward propogation
    a1 = [1, X(i,:)]; % 1 by 401
    z2 = a1 * Theta1'; % 1 by 25
    a2 = [1, sigmoid(z2)]; % 1 by 26
    z3 = a2 * Theta2'; % 1 by 10
    a3 = sigmoid(z3); % output doesn't need bias unit, 1 by 10
    % computer delta
    yk = y_mod(i,:); % 1 by 10
    delta_3 = a3 - yk; % 1 by 10
    delta_2 = delta_3 * Theta2 .* [0, sigmoidGradient(z2)]; % 1 by 26
    DELTA_1 = DELTA_1 + delta_2(2:end)' * a1; % 25 by 401, with regularization
    DELTA_2 = DELTA_2 + delta_3' * a2; % 10 by 26, with regularization
end

% obtain the gradient of cost function w.r.t Theta1 and Theta2
Theta1_grad = DELTA_1/m + lambda/m*[zeros(hidden_layer_size,1) Theta1(:,2:end)];
Theta2_grad = DELTA_2/m + lambda/m*[zeros(num_labels,1) Theta2(:,2:end)];
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
