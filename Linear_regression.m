% Linear regression in Octave. From the Coursera Machine Learning course.

clear ; 
close all; 
clc


% Here is a little toy example of what we're doing

% We create 4 toy training example pairs [time ; value] 
toy_data = [ 0 0 ; 
             1 3 ; 
             3 2 ; 
             6 7];

% X is the time values 
X_toy = toy_data(:,1);
% Y is the values
y_toy = toy_data(:,2);

%take a look at it
plot(X_toy, y_toy, 'bx', 'MarkerSize',10) % plot the scatter data 
ylabel('Time');            % Set the y axis label
xlabel('Observation'); % Set the x axis label

% Set up a gradient descent 
m_t = length(y_toy); % number of examples
X_t = [ones(m_t,1), toy_data(:,1)]; % we add a column of ones to set up the matrix 
theta_t = zeros(2,1); % This is our fit parameters theta_0 and theta_1

%What is the cost with these parameters???

sum((X_t * theta_t - y_toy).^2) / ( 2 * m_t)

% ie 
sum(([1 0; 1 1; 1 3; 1 6]*[0; 0] - [0; 3; 2; 7]).^2) / (2* m_t)

% minimise the cost using gradient descent 
its = 1000; % number of iterations 
alp = 0.015; % learning rate 
for iter = 1:its
theta_t = theta_t - (alp / m_t) .* sum((X_t * theta_t - y_toy).*X_t)';
end 
% Plot the linear fit
hold on; % keep previous plot visible
plot(X_t(:,2), X_t*theta_t, '-')
hold off % don't overlay any more plots on this figure









% Let's try with some more training examples 
%Load the training data 
train_data = load('ex1data1.txt');

%plot the data 
figure;
X = train_data(:, 1); %isolate the first column. 
y = train_data(:, 2); %isolate the second column. 
plot(X, y, 'rx', 'MarkerSize',10) % plot the scatter data 
ylabel('Profit in $10,000s');            % Set the y axis label
xlabel('Population of City in 10,000s'); % Set the x axis label

%Start gradient descent
m = length(y); % m is the number of training examples
X = [ones(m, 1), train_data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters at 0 0 

iterations = 5000; % our iteration parameter
alpha = 0.01; % learning speed 

J = 0; % initialise the cost
J_history = zeros(iterations, 1);

for iter = 1:iterations
theta = theta - (alpha / m) .* sum((X * theta - y).*X)';
J_history(iter) = costComputer(X, y, theta, m);
end 

% print out our theta to screen
fprintf('Theta found by gradient descent: ');
fprintf('%f %f \n', theta(1), theta(2));

% Plot the linear fit
hold on; % keep previous plot visible
plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure

% Some predictions for population sizes of 35,000 and 70,000
predict1 = [1, 3.5] *theta;
fprintf('For population = 35,000, we predict a profit of %f\n',...
    predict1*10000);
predict2 = [1, 7] * theta;
fprintf('For population = 70,000, we predict a profit of %f\n',...
    predict2*10000);
    
% Visualing the cost function    
% Grid of values for J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% Create initial J_vals matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Loop over the values in our grid and calculate the cost J
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];    
	  J_vals(i,j) = costComputer(X, y, t, m);
    end
end

% Because of the way meshgrids work in the surf command, we need to 
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Plot a 3D surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); 
ylabel('\theta_1'); 
zlabel('Cost function J')
hold on 

% Plot the theta value we found on the surface plot 
plot(theta(1), theta(2), costComputer(X, y, [theta(1); theta(2)], m), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 60))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;

%Plot the theta value we found on the contour plot 
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);


%Let's look at the convergence of our cost function
its=[1:iterations];
figure;
plot(its,J_history, '-');
xlabel('Iteration'); 
ylabel('J'); 
