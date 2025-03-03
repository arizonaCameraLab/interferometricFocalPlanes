clear; clc;
syms d1 theta1 I1 a real

% Define the PSF function
psf = @(x) exp(-x.^2 / 2);

% Define x and y coordinates for both sets of parameters
x11 = d1 - a * cos(theta1); x21 = d1 + a * cos(theta1);
y11 = -a * sin(theta1); y21 = a * sin(theta1);
x31 = d1 - a * cos(pi/2 - theta1); x41 = d1 + a * cos(pi/2 - theta1);
y31 = a * sin(pi/2 - theta1); y41 = -a * sin(pi/2 - theta1);

% Define functions A, B, C
A1 = I1 * psf(x11).^2;
B1 = I1 * psf(x21).^2;
C1 = I1 * psf(x11) .* psf(x21) .* psf((y11 - y21)/sqrt(2));

A2 = I1 * psf(x31).^2;
B2 = I1 * psf(x41).^2;
C2 = I1 * psf(x31) .* psf(x41) .* psf((y31 - y41)/sqrt(2));

% Define the function vector
F = [A1; B1; C1; A2; B2; C2];

% Compute the Jacobian matrix J_F = dF/d[d1, d2, theta1, theta2, I1, I2]
vars = [d1, theta1, I1];
J_F = jacobian(F, vars);

% Convert Jacobian matrix to numerical function for faster evaluation
J_F_func = matlabFunction(J_F, 'Vars', {d1, theta1, I1, a});

% Display symbolic Jacobian matrix
disp('Jacobian Matrix:');
disp(J_F);

% Example numerical substitution
d1_val = 0.1;
theta1_val = pi/3;
I1_val = 2.0;
a_val = 0.5;

% Evaluate Jacobian numerically
J_num = double(J_F_func(d1_val, theta1_val, I1_val, a_val));

% Display numerical Jacobian
disp('Numerical Jacobian:');
disp(J_num);
disp(cond(J_num))

I_num = J_num([1:2,4:5], :);
disp(I_num);
disp(cond(I_num))

%%
clc, clear, close all
syms d1 theta1 I1 a real

% Define the PSF function
psf = @(x) exp(-x.^2 / 2);

% Define x and y coordinates for both sets of parameters
x11 = d1 - a * cos(theta1); x21 = d1 + a * cos(theta1);
y11 = -a * sin(theta1); y21 = a * sin(theta1);
x31 = d1 - a * cos(pi/2 - theta1); x41 = d1 + a * cos(pi/2 - theta1);
y31 = a * sin(pi/2 - theta1); y41 = -a * sin(pi/2 - theta1);

% Define functions A, B, C
A1 = I1 * psf(x11).^2;
B1 = I1 * psf(x21).^2;
C1 = I1 * psf(x11) .* psf(x21) .* psf((y11 - y21)/sqrt(2));

A2 = I1 * psf(x31).^2;
B2 = I1 * psf(x41).^2;
C2 = I1 * psf(x31) .* psf(x41) .* psf((y31 - y41)/sqrt(2));

% Define the function vector
F = [A1; B1; C1; A2; B2; C2];

% Compute the Jacobian matrix J_F = dF/d[d1, theta1, I1]
vars = [d1, theta1, I1];
J_F = jacobian(F, vars);

% Convert Jacobian matrix to numerical function for faster evaluation
J_F_func = matlabFunction(J_F, 'Vars', {d1, theta1, I1, a});

% Define scanning parameters
d_vals = linspace(0, 1, 100);      % Scan d1 from 0 to 1
theta_vals = linspace(0, pi/2, 100); % Scan theta1 from 0 to pi/2
a_val = 0.5;
I1_val = 2.0;

% Initialize condition number storage
cond_J = zeros(length(d_vals), length(theta_vals));
cond_I = zeros(length(d_vals), length(theta_vals));

% Compute condition numbers over the grid
for i = 1:length(d_vals)
    for j = 1:length(theta_vals)
        d1_val = d_vals(i);
        theta1_val = theta_vals(j);
        
        % Evaluate Jacobian numerically
        J_num = double(J_F_func(d1_val, theta1_val, I1_val, a_val));
        
        % Compute condition numbers
        cond_J(i, j) = cond(J_num);
        I_num = J_num([1:2,4:5], :); % Extract specific rows
        cond_I(i, j) = cond(I_num);
    end
end

% Plot condition number for J_F
figure;
imagesc(theta_vals, d_vals, log10(cond_J)); % Log scale for better visualization
colormap('jet');
colorbar;
xlabel('\theta_1 (rad)');
ylabel('d_1');
title('log_{10}(cond(J_F))');

% Plot condition number for I_num
figure;
imagesc(theta_vals, d_vals, log10(cond_I)); % Log scale for better visualization
colormap('jet');
colorbar;
xlabel('\theta_1 (rad)');
ylabel('d_1');
title('log_{10}(cond(I_num))');

% Plot the difference between cond(J_F) and cond(I_num)
figure;
diff_cond = log10(cond_I) - log10(cond_J);
imagesc(theta_vals, d_vals, diff_cond);
colormap('jet');
colorbar;

% Add contour line for zero boundary
hold on;
contour(theta_vals, d_vals, diff_cond, [0 0], 'k', 'LineWidth', 2);
hold off;

xlabel('\theta_1 (rad)');
ylabel('d_1');
title('log_{10}(cond(I_num)) - log_{10}(cond(J_F))');

