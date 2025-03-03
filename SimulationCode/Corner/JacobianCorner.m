clear; clc;
syms x0 theta I a real

% Define x and y coordinates (x0 = y0)
y0 = x0;

x1 = x0 - a * cos(theta);
x2 = x0 + a * cos(theta);
y1 = y0 - a * sin(theta);
y2 = y0 + a * sin(theta);
x3 = x0 - a * cos(pi/2 - theta);
x4 = x0 + a * cos(pi/2 - theta);
y3 = y0 + a * sin(pi/2 - theta);
y4 = y0 - a * sin(pi/2 - theta);

% Define functions A, B, C
A12 = I * pi/4 * (1 + erf(x1));
B12 = I * pi/4 * (1 + erf(x2));
C12 = I * pi/4 * exp(-((y1 - y2)^2)/4 - ((x1 - x2)^2)/4) ...
    * (1 + erf((x1 + x2) / 2)) * (1 + erf((y1 + y2) / 2));

A34 = I * pi/4 * (1 + erf(x3));
B34 = I * pi/4 * (1 + erf(x4));
C34 = I * pi/4 * exp(-((y3 - y4)^2)/4 - ((x3 - x4)^2)/4) ...
    * (1 + erf((x3 + x4) / 2)) * (1 + erf((y3 + y4) / 2));

% Define the function vector
F = [A12; B12; C12; A34; B34; C34];

% Compute the Jacobian matrix J_F = dF/d[x0, theta, I]
vars = [x0, theta, I];
J_F = jacobian(F, vars);

% Convert Jacobian matrix to numerical function for speed
J_F_func = matlabFunction(J_F, 'Vars', {x0, theta, I, a});

% Define range for x0 and theta
x0_vals = linspace(-1, 1, 100); % Scanning x0 in [-1,1]
theta_vals = linspace(0, pi/2, 50); % Scanning theta in [0, pi/2]
a_val = 0.5;
I_val = 2.0;

% Initialize matrices to store condition numbers
cond_J = zeros(length(x0_vals), length(theta_vals));
cond_J_reduced = zeros(length(x0_vals), length(theta_vals));

% Compute condition numbers over the grid
for i = 1:length(x0_vals)
    for j = 1:length(theta_vals)
        x0_val = x0_vals(i);
        theta_val = theta_vals(j);
        
        % Evaluate Jacobian numerically using the precomputed function
        J_num = J_F_func(x0_val, theta_val, I_val, a_val);
        
        % Compute condition numbers
        cond_J(i, j) = cond(J_num);
        cond_J_reduced(i, j) = cond(J_num([1:2, 4:5], :));
    end
end

%% **Plot**
figure;
imagesc(theta_vals, x0_vals, log10(cond_J)); % Log scale for better visualization
colormap('jet'); % Red for high cond, Blue for low cond
colorbar;
xlabel('\theta (rad)');
ylabel('x_0 (y_0)');
title('log_{10}(cond(J_F))');

figure;
imagesc(theta_vals, x0_vals, log10(cond_J_reduced));
colormap('jet'); % Red for high cond, Blue for low cond
colorbar;
xlabel('\theta (rad)');
ylabel('x_0 (y_0)');
title('log_{10}(cond(J_F([1:2,4:5],:)))');

figure;
diff_cond = log10(cond_J_reduced) - log10(cond_J);
imagesc(theta_vals, x0_vals, diff_cond);
colormap('jet'); 
colorbar;

% Add contour line for zero boundary
hold on;
contour(theta_vals, x0_vals, diff_cond, [0 0], 'k', 'LineWidth', 2);
hold off;

xlabel('\theta (rad)');
ylabel('x_0 (y_0)');
title('log_{10}(cond(J_F([1:2,4:5],:))) - log_{10}(cond(J_F))');
