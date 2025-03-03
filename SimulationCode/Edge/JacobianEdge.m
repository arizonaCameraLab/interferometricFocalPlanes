clear; clc; close all
syms d theta I a real

% Define x and y coordinates
x1 = d - a * cos(theta);
x2 = d + a * cos(theta);
y1 = -a * sin(theta);
y2 = a * sin(theta);
x3 = d - a * cos(pi/2 - theta);
x4 = d + a * cos(pi/2 - theta);
y3 = -a * sin(pi/2 - theta);
y4 = a * sin(pi/2 - theta);

% Define functions A, B, C
A12 = I * sqrt(pi)/2 * (1 + erf(x1 / sqrt(2)));
B12 = I * sqrt(pi)/2 * (1 + erf(x2 / sqrt(2)));
C12 = I * sqrt(pi)/2 * exp(-((y1 - y2)^2)/4 - ((x1 - x2)^2)/4) * (1 + erf((x1 + x2) / (2 * sqrt(2))));

A34 = I * sqrt(pi)/2 * (1 + erf(x3 / sqrt(2)));
B34 = I * sqrt(pi)/2 * (1 + erf(x4 / sqrt(2)));
C34 = I * sqrt(pi)/2 * exp(-((y3 - y4)^2)/4 - ((x3 - x4)^2)/4) * (1 + erf((x3 + x4) / (2 * sqrt(2))));

% Define the function vector
F = [A12; B12; C12; A34; B34; C34];

% Compute the Jacobian matrix J_F = dF/d(d,theta,I)
vars = [d, theta, I];
J_F = jacobian(F, vars);

% Convert Jacobian matrix to numerical function for faster evaluation
J_F_func = matlabFunction(J_F, 'Vars', {d, theta, I, a});

% Define range for d and theta
d_vals = linspace(-1, 1, 200);
theta_vals = linspace(0, pi/2, 100);
a_val = 0.5;
I_val = 2.0;

% Initialize matrices to store condition numbers
cond_J = zeros(length(d_vals), length(theta_vals));
cond_J_reduced = zeros(length(d_vals), length(theta_vals));

% Compute condition numbers over the grid
for i = 1:length(d_vals)
    for j = 1:length(theta_vals)
        d_val = d_vals(i);
        theta_val = theta_vals(j);
        
        % Evaluate Jacobian numerically using the precomputed function
        J_num = J_F_func(d_val, theta_val, I_val, a_val);
        
        % Compute condition numbers
        cond_J(i, j) = cond(J_num);
        cond_J_reduced(i, j) = cond(J_num([1:2, 4:5], :));
    end
end

% Compute the difference in log condition numbers
log_diff = log10(cond_J_reduced) - log10(cond_J);

% Plot the condition number for J_F
figure;
imagesc(theta_vals, d_vals, log10(cond_J)); % log scale for better visualization
colorbar; colormap("jet")
xlabel('\theta (rad)');
ylabel('d');
title('log_{10}(cond(J_F))');

% Plot the condition number for reduced J_F
figure;
imagesc(theta_vals, d_vals, log10(cond_J_reduced));
colorbar; colormap("jet")
xlabel('\theta (rad)');
ylabel('d');
title('log_{10}(cond(J_F([1:2,5:6],:)))');

% Plot the condition number ratio with highlight for log_diff > 0
figure;
hold on;
imagesc(theta_vals, d_vals, log_diff);
contour(theta_vals, d_vals, log_diff, [0 0], 'r', 'LineWidth', 2); % Highlight zero crossing
colorbar; colormap("jet")
xlabel('\theta (rad)');
ylabel('d');
title('log_{10}(cond(J_F([1:2,3:4],:))) - log_{10}(cond(J_F))');
hold off;


