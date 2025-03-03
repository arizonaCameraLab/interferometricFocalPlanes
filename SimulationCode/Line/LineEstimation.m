clear; clc; close all;
[G1_func, G2_func, G3_func, G4_func] = MeasureLine();

% Define d_true and theta_true range
d_values = linspace(0.01, 0.3, 8); % 5 values between 0.05 and 0.3
theta_values = linspace(0, pi/2, 8); % 5 values between 0.001 and pi/2

% Simulation Parameters
num_trials = 2000; % Number of Monte Carlo trials
total_measurements = 6;  % Number of measurements per trial
total_photons = 1000000; % Total photons available for measurements
photons = total_photons / total_measurements; % Photons allocated per measurement
a_val = 1;
I_true = 1;
% Define a function to add Poisson noise to measurement data
addPoisson = @(data, scale) poissrnd(data .* scale) ./ scale;

% Storage for standard deviation results
std_results_fixed = zeros(length(d_values), length(theta_values), 3);
std_results_adaptive = zeros(length(d_values), length(theta_values), 3);
std_results_intensity = zeros(length(d_values), length(theta_values), 3);
rand_ratio = 0.0;  % Random perturbation ratio for optimization initialization

tic
% Loop over all d_true and theta_true values
for d_idx = 1:length(d_values)
    for theta_idx = 1:length(theta_values)
        d_true = d_values(d_idx);
        theta_true = theta_values(theta_idx);
        
        % Store estimated parameters for all trials
        results_fixed = zeros(num_trials, 3);
        results_adaptive = zeros(num_trials, 3);
        results_intensity = zeros(num_trials, 3);
        % Use parallel processing to speed up Monte Carlo trials
        parfor trial = 1:num_trials
            % ------ Fixed Measurement Strategy ------
            % Fixed measurment: Given operator H with angle:alpha to
            % operate on J:Mutual Intensity matrix 
            alpha_vals_fixed = linspace(-pi/2, pi/2, total_measurements);
            G1_meas_fixed = addPoisson(G1_func(d_true, theta_true, I_true, a_val, alpha_vals_fixed), photons);
            G2_meas_fixed = addPoisson(G2_func(d_true, theta_true, I_true, a_val, alpha_vals_fixed), photons);
            G3_meas_fixed = addPoisson(G3_func(d_true, theta_true, I_true, a_val, alpha_vals_fixed), photons);
            G4_meas_fixed = addPoisson(G4_func(d_true, theta_true, I_true, a_val, alpha_vals_fixed), photons);
        
            obj_fun_fixed = @(params) -sum(G1_meas_fixed .* log(G1_func(params(1), params(2), params(3), a_val, alpha_vals_fixed)) - ...
                                            G1_func(params(1), params(2), params(3), a_val, alpha_vals_fixed)) ...
                                      -sum(G2_meas_fixed .* log(G2_func(params(1), params(2), params(3), a_val, alpha_vals_fixed)) - ...
                                            G2_func(params(1), params(2), params(3), a_val, alpha_vals_fixed)) ...
                                      -sum(G3_meas_fixed .* log(G3_func(params(1), params(2), params(3), a_val, alpha_vals_fixed)) - ...
                                            G3_func(params(1), params(2), params(3), a_val, alpha_vals_fixed)) ...
                                      -sum(G4_meas_fixed .* log(G4_func(params(1), params(2), params(3), a_val, alpha_vals_fixed)) - ...
                                            G4_func(params(1), params(2), params(3), a_val, alpha_vals_fixed));
        
            results_fixed(trial, :) = fminunc(obj_fun_fixed, [d_true, theta_true, I_true]+ rand_ratio * randn(1, 3));
            % Adaptive
        
            % ------ Adaptive Measurement Strategy ------
            % Adaptive measurments: diagonizing the measurments for
            % total_measurements steps to find the optimal alpha

            alpha1_vals = []; alpha2_vals = [];
            G1_adaptive = []; G2_adaptive = []; G3_adaptive = []; G4_adaptive = [];
            params_est = [d_true, theta_true, I_true];
            
            for step = 1:total_measurements
                alpha_candidates = linspace(0, pi/2, 101);
                if step == 1
                    alpha1_next = 0;
                    alpha2_next = 0;
                else
                    % alpha1 for G1, G2
                    range_12 = arrayfun(@(alpha) range([G1_func(params_est(1), params_est(2), params_est(3), a_val, alpha); 
                                                        G2_func(params_est(1), params_est(2), params_est(3), a_val, alpha)]), alpha_candidates);
                    [~, idx1] = max(range_12); alpha1_next = alpha_candidates(idx1);
            
                    % alpha2 for G3, G4
                    range_34 = arrayfun(@(alpha) range([G3_func(params_est(1), params_est(2), params_est(3), a_val, alpha); 
                                                        G4_func(params_est(1), params_est(2), params_est(3), a_val, alpha)]), alpha_candidates);
                    [~, idx2] = max(range_34); alpha2_next = alpha_candidates(idx2);
                end
            
                alpha1_vals = [alpha1_vals, alpha1_next]; alpha2_vals = [alpha2_vals, alpha2_next];
                G1_adaptive = [G1_adaptive, addPoisson(G1_func(d_true, theta_true, I_true, a_val, alpha1_next), photons)];
                G2_adaptive = [G2_adaptive, addPoisson(G2_func(d_true, theta_true, I_true, a_val, alpha1_next), photons)];
                G3_adaptive = [G3_adaptive, addPoisson(G3_func(d_true, theta_true, I_true, a_val, alpha2_next), photons)];
                G4_adaptive = [G4_adaptive, addPoisson(G4_func(d_true, theta_true, I_true, a_val, alpha2_next), photons)];
            
                % use maximum likelihood method to estimate
                obj_fun_adaptive = @(params) -sum(G1_adaptive .* log(G1_func(params(1), params(2), params(3), a_val, alpha1_vals)) - ...
                                                  G1_func(params(1), params(2), params(3), a_val, alpha1_vals)) ...
                                            -sum(G2_adaptive .* log(G2_func(params(1), params(2), params(3), a_val, alpha1_vals)) - ...
                                                  G2_func(params(1), params(2), params(3), a_val, alpha1_vals)) ...
                                            -sum(G3_adaptive .* log(G3_func(params(1), params(2), params(3), a_val, alpha2_vals)) - ...
                                                  G3_func(params(1), params(2), params(3), a_val, alpha2_vals)) ...
                                            -sum(G4_adaptive .* log(G4_func(params(1), params(2), params(3), a_val, alpha2_vals)) - ...
                                                  G4_func(params(1), params(2), params(3), a_val, alpha2_vals));
            
                params_est = fminunc(obj_fun_adaptive, [d_true, theta_true, I_true]+ rand_ratio * randn(1, 3));
            end
            
            results_adaptive(trial, :) = params_est;
            %            % ------ Intensity Measurement Strategy ------
            % Intensity measurments: meausre the intensity for each sensor
            % points
            alpha_vals_intensity = -pi/2 * zeros(1, total_measurements);
            G1_meas_intensity = addPoisson(G1_func(d_true, theta_true, I_true, a_val, alpha_vals_intensity), photons);
            G2_meas_intensity = addPoisson(G2_func(d_true, theta_true, I_true, a_val, alpha_vals_intensity), photons);
            G3_meas_intensity = addPoisson(G3_func(d_true, theta_true, I_true, a_val, alpha_vals_intensity), photons);
            G4_meas_intensity = addPoisson(G4_func(d_true, theta_true, I_true, a_val, alpha_vals_intensity), photons);
        
            obj_fun_intensity = @(params) -sum(G1_meas_intensity .* log(G1_func(params(1), params(2), params(3), a_val, alpha_vals_intensity)) - ...
                                               G1_func(params(1), params(2), params(3), a_val, alpha_vals_intensity)) ...
                                         -sum(G2_meas_intensity .* log(G2_func(params(1), params(2), params(3), a_val, alpha_vals_intensity)) - ...
                                               G2_func(params(1), params(2), params(3), a_val, alpha_vals_intensity)) ...
                                         -sum(G3_meas_intensity .* log(G3_func(params(1), params(2), params(3), a_val, alpha_vals_intensity)) - ...
                                               G3_func(params(1), params(2), params(3), a_val, alpha_vals_intensity)) ...
                                         -sum(G4_meas_intensity .* log(G4_func(params(1), params(2), params(3), a_val, alpha_vals_intensity)) - ...
                                               G4_func(params(1), params(2), params(3), a_val, alpha_vals_intensity));
        
            results_intensity(trial, :) = fminunc(obj_fun_intensity, [d_true, theta_true, I_true]+ rand_ratio * randn(1, 3));
        end


        for param_idx = 1:3
            std_results_fixed(d_idx, theta_idx, param_idx) = std(results_fixed(:, param_idx));
            std_results_adaptive(d_idx, theta_idx, param_idx) = std(results_adaptive(:, param_idx));
            std_results_intensity(d_idx, theta_idx, param_idx) = std(results_intensity(:, param_idx));
        end
    end
end
toc
%%
save("LineEstimation")
%% Load and visualize results (all results)
% clear; clc;close all; load("LineEstimation.mat");
% 
% theta_val_slice = [0.1, 0.7, 1.3];
% d_val_slice = [0.05, 0.1, 0.2, 0.3];
% 
% 
% % Find closest indices for theta_val_slice and d_val_slice
% [~, theta_idx_slice] = min(abs(theta_values' - theta_val_slice), [], 1);
% [~, d_idx_slice] = min(abs(d_values' - d_val_slice), [], 1);
% 
% % Parameter names
% param_names = {'d', '\theta', 'I'};
% % param_names = {'\theta'};
% % Loop through each estimated parameter
% for param_idx = 1:3
%     figure;
%     set(gcf, 'Position', [100, 100, 1000, 600]); 
%     set(gca, 'FontSize', 11)
%     % Plot sigma vs. d for fixed theta values
%     for subplot_idx = 1:3
%         theta_idx = theta_idx_slice(subplot_idx);
%         subplot(2, 3, subplot_idx);
%         hold on;
% 
%         % Extract sigma values for different strategies
%         sigma_fixed = squeeze(std_results_fixed(:, theta_idx, param_idx));
%         sigma_adaptive = squeeze(std_results_adaptive(:, theta_idx, param_idx));
%         sigma_intensity = squeeze(std_results_intensity(:, theta_idx, param_idx));
% 
%         % Plot sigma vs. d
%         plot(d_values, sigma_intensity, '-b', 'LineWidth', 1.5, 'DisplayName', 'Intensity');hold on;
%         plot(d_values, sigma_fixed, '-r', 'LineWidth', 1.5, 'DisplayName', 'Fixed');hold on;
%         plot(d_values, sigma_adaptive, '-g', 'LineWidth', 1.5, 'DisplayName', 'Adaptive');
% 
% 
%         xlabel('d values');
%         ylabel(['\sigma_{', param_names{param_idx}, '}']);
%         title(['\theta \approx ', num2str(rad2deg(theta_values(theta_idx)), '%.2f') char(176)]);
%         legend show;
%         grid on;
%     end
% 
%     % Plot sigma vs. theta for fixed d values
%     for subplot_idx = 1:3
%         d_idx = d_idx_slice(subplot_idx);
%         subplot(2, 3, subplot_idx + 3);
%         hold on;
% 
%         % Extract sigma values for different strategies
%         sigma_fixed = squeeze(std_results_fixed(d_idx, :, param_idx));
%         sigma_adaptive = squeeze(std_results_adaptive(d_idx, :, param_idx));
%         sigma_intensity = squeeze(std_results_intensity(d_idx, :, param_idx));
% 
%         % Plot sigma vs. theta
%         plot(rad2deg(theta_values), sigma_intensity, '-b', 'LineWidth', 1.5, 'DisplayName', 'Intensity');hold on;
%         plot(rad2deg(theta_values), sigma_fixed, '-r', 'LineWidth', 1.5, 'DisplayName', 'Fixed');hold on;
%         plot(rad2deg(theta_values), sigma_adaptive, '-g', 'LineWidth', 1.5, 'DisplayName', 'Adaptive');
% 
% 
%         xlabel('\theta values');
%         ylabel(['\sigma_{', param_names{param_idx}, '}']);
%         title(['d \approx ', num2str(d_values(d_idx), '%.2f')]);
%         legend show;
%         grid on;
%     end
% end

%%
% plot resutls on paper
clear; clc;close all; load("LineEstimation.mat");

theta_val_slice = [0.1, 0.7, 1.3];
d_val_slice = [0.05, 0.1, 0.2, 0.3];


% Find closest indices for theta_val_slice and d_val_slice
[~, theta_idx_slice] = min(abs(theta_values' - theta_val_slice), [], 1);
[~, d_idx_slice] = min(abs(d_values' - d_val_slice), [], 1);

% Parameter names
param_names = {'\rho', '\theta', 'I'};
% param_names = {'\theta'};
% Loop through each estimated parameter
for param_idx = 2
    figure;
    set(gcf, 'Position', [100, 100, 1000, 280]); 
    set(gca, 'FontSize', 11)
    % Plot sigma vs. d for fixed theta values
    for subplot_idx = 1:3
        theta_idx = theta_idx_slice(subplot_idx);
        subplot(1, 3, subplot_idx);
        hold on;

        % Extract sigma values for different strategies
        sigma_fixed = squeeze(std_results_fixed(:, theta_idx, param_idx));
        sigma_adaptive = squeeze(std_results_adaptive(:, theta_idx, param_idx));
        sigma_intensity = squeeze(std_results_intensity(:, theta_idx, param_idx));

        % Plot sigma vs. d
    plot(d_values, sigma_intensity, '-b', 'LineWidth', 1.5, 'DisplayName', 'Intensity'); hold on;
    plot(d_values, sigma_fixed, '-r', 'LineWidth', 1.5, 'DisplayName', 'Fixed'); hold on;
    plot(d_values, sigma_adaptive, '-g', 'LineWidth', 1.5, 'DisplayName', 'Adaptive');

    xlabel('\rho ', 'FontSize', 15);
    ylabel(['Estimation Error of {', param_names{param_idx}, '}'], 'FontSize', 15);
    % title(['\theta \approx ', num2str(rad2deg(theta_values(theta_idx)), '%.2f') char(176)], 'FontSize', 17);

    if subplot_idx == 1
        lgd = legend('show');
        set(lgd, 'FontSize', 12); % Set legend font size
    end

    grid on;

    end

    % % Plot sigma vs. theta for fixed d values
    % for subplot_idx = 1:3
    %     d_idx = d_idx_slice(subplot_idx);
    %     subplot(2, 3, subplot_idx + 3);
    %     hold on;
    % 
    %     % Extract sigma values for different strategies
    %     sigma_fixed = squeeze(std_results_fixed(d_idx, :, param_idx));
    %     sigma_adaptive = squeeze(std_results_adaptive(d_idx, :, param_idx));
    %     sigma_intensity = squeeze(std_results_intensity(d_idx, :, param_idx));
    % 
    %     % Plot sigma vs. theta
    %     plot(rad2deg(theta_values), sigma_intensity, '-b', 'LineWidth', 1.5, 'DisplayName', 'Intensity');hold on;
    %     plot(rad2deg(theta_values), sigma_fixed, '-r', 'LineWidth', 1.5, 'DisplayName', 'Fixed');hold on;
    %     plot(rad2deg(theta_values), sigma_adaptive, '-g', 'LineWidth', 1.5, 'DisplayName', 'Adaptive');
    % 
    % 
    %     xlabel('\theta values');
    %     ylabel(['\sigma_{', param_names{param_idx}, '}']);
    %     title(['d \approx ', num2str(d_values(d_idx), '%.2f')]);
    %     legend show;
    %     grid on;
    % end
end

