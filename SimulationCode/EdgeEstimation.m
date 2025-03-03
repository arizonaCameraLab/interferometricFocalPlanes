clear; clc; close all;  % Clear workspace, command window, and close all figures

% Start parallel pool (if not already running) to speed up computation
if isempty(gcp('nocreate'))
    parpool;  % Start parallel processing workers
end

% Define measurement function to simulate edge detection
measure = @(d, theta, I, a, alpha1, alpha2) MeasureEdge(d, theta, I, a, alpha1, alpha2);

% Simulation Parameters
num_trials = 2000;  % Number of Monte Carlo trials
total_measurements = 6;  % Number of measurements per trial
total_photons = 1000000;  % Total photons available for measurements
photons = total_photons / total_measurements;  % Photons allocated per measurement
a_val = 1;  % Sensor pair half distance for diagonal and anti-diagonal pair
I_true = 1;  % True intensity value

% Define a function to add Poisson noise to measurement data
addPoisson = @(data, scale) poissrnd(data .* scale) ./ scale;

% Define the range of true edge parameters d (distance) and theta (angle)
d_values = linspace(0.05, 0.3, 5);  % 5 values between 0.05 and 0.3
theta_values = linspace(0.001, pi/2, 5);  % 5 values between 0.001 and pi/2

% Storage for standard deviation results (3 parameters: d, theta, I)
std_results_fixed = zeros(length(d_values), length(theta_values), 3);
std_results_adaptive = zeros(length(d_values), length(theta_values), 3);
std_results_intensity = zeros(length(d_values), length(theta_values), 3);
rand_ratio = 0.0;  % Random perturbation ratio for optimization initialization

tic  % Start timing execution

% Loop over different true values of d and theta
for d_idx = 1:length(d_values)
    for theta_idx = 1:length(theta_values)
        d_true = d_values(d_idx);  % Set true distance
        theta_true = theta_values(theta_idx);  % Set true angle

        % Initialize storage for estimated parameters for all trials
        results_fixed = zeros(num_trials, 3);
        results_adaptive = zeros(num_trials, 3);
        results_intensity = zeros(num_trials, 3);

        % Use parallel processing to speed up Monte Carlo trials
        for trial = 1:num_trials

            % ------ Intensity Measurement Strategy ------
            % Intensity measurments: meausre the intensity for each sensor
            % points
            alpha1_intensity = zeros(total_measurements, 1);
            alpha2_intensity = zeros(total_measurements, 1);
            G_intensity = measure(d_true, theta_true, I_true, a_val, alpha1_intensity, alpha2_intensity);

            % G_intensity = measure(d_true, theta_true, I_true, a_val, 0, 0);  % Single intensity measurement
            G_meas_intensity = addPoisson(G_intensity, photons);  % Add Poisson noise
            
            obj_fun_intensity = @(params) sum(measure(params(1), params(2), params(3), a_val, alpha1_intensity, alpha1_intensity) ...
                                            - G_meas_intensity .* log(measure(params(1), params(2), params(3), a_val, alpha1_intensity, alpha1_intensity)) );
            results_intensity(trial, :) = fminunc(obj_fun_intensity, [d_true, theta_true, I_true] + rand_ratio * randn(1, 3));

            % ------ Fixed Measurement Strategy ------
            % Fixed measurment: Given operator H with angle:alpha to
            % operate on J:Mutual Intensity matrix 
            alpha_vals_fixed = linspace(-pi/2, pi/2, total_measurements);  % Fixed measurement angles
            G_fixed = measure(d_true, theta_true, I_true, a_val, alpha_vals_fixed, alpha_vals_fixed);  % Simulated measurement
            G_meas_fixed = addPoisson(G_fixed, photons);  % Add Poisson noise
            
            % Define objective function for optimization
            obj_fun_fixed = @(params) sum(measure(params(1), params(2), params(3), a_val, alpha_vals_fixed, alpha_vals_fixed) ...
                                      - G_meas_fixed .* log(measure(params(1), params(2), params(3), a_val, alpha_vals_fixed, alpha_vals_fixed)));
            % Optimize parameters using fminunc
            results_fixed(trial, :) = fminunc(obj_fun_fixed, [d_true, theta_true, I_true] + rand_ratio * randn(1, 3));

            % ------ Adaptive Measurement Strategy ------
            % Adaptive measurments: diagonizing the measurments for
            % total_measurements steps to find the optimal alpha

            alpha1_adaptive = []; % potential adative steps for 1 pair sensor  
            alpha2_adaptive = [];
            G_meas_adaptive = [];
            params_est = [d_true, theta_true, I_true];  % Initial parameter estimate

            for step = 1:total_measurements
                if step == 1 % First step is 0 to measure diagonal elements: Intensity
                    alpha1_next = 0; alpha2_next = 0;
                else
                    % Find optimal angles maximizing measurement variation
                    % which would diagoinalize the matrix J
                    alpha_candidates = linspace(0, pi/2, 100);
                    [~, idx1] = max(arrayfun(@(a) range(measure(params_est(1), params_est(2), params_est(3), a_val, a, alpha2_adaptive)), alpha_candidates));
                    alpha1_next = alpha_candidates(idx1);
                    [~, idx2] = max(arrayfun(@(a) range(measure(params_est(1), params_est(2), params_est(3), a_val, alpha1_adaptive, a)), alpha_candidates));
                    alpha2_next = alpha_candidates(idx2);
                end

                % Update adaptive measurement angles alpha
                alpha1_adaptive = [alpha1_adaptive; alpha1_next];
                alpha2_adaptive = [alpha2_adaptive; alpha2_next];
                new_measure = addPoisson(measure(d_true, theta_true, I_true, a_val, alpha1_next, alpha2_next), photons);
                % G_meas_adaptive = [G_meas_adaptive; new_measure];
                if step == 1
                    G_meas_adaptive = new_measure;
                else
                    old_g = reshape(G_meas_adaptive, [], 4);
                    G_meas_adaptive = reshape([old_g; new_measure(:)'], [], 1);
                end
                
                % Optimize parameters using fminunc with random
                % perturbation
                obj_fun_adaptive = @(params) sum(measure(params(1), params(2), params(3), a_val, alpha1_adaptive, alpha2_adaptive) ...
                                                - G_meas_adaptive .* log(measure(params(1), params(2), params(3), a_val, alpha1_adaptive, alpha2_adaptive)) );
                params_est = fminunc(obj_fun_adaptive, [d_true, theta_true, I_true] + rand_ratio * randn(1, 3));
            end
            results_adaptive(trial, :) = params_est;
        end

        % Apply 3-sigma filtering before computing standard deviations
        for param_idx = 1:3
            std_results_fixed(d_idx, theta_idx, param_idx) = std(results_fixed(:, param_idx));
            std_results_adaptive(d_idx, theta_idx, param_idx) = std(results_adaptive(:, param_idx));
            std_results_intensity(d_idx, theta_idx, param_idx) = std(results_intensity(:, param_idx));
        end
    end
end
toc  % End timing execution

%% Save all results in a file for future analysis
file_name = "EdgeEstimation_results";
save(file_name);

%% Load and visualize results (all results)
% clear; close all; load("EdgeEstimation_results.mat");
% 
% theta_val_slice = [0.1, 0.3, 0.7];
% d_val_slice = [0.05, 0.1, 0.2, 0.3];
% 
% 
% % Find closest indices for theta_val_slice and d_val_slice
% [~, theta_idx_slice] = min(abs(theta_values' - theta_val_slice), [], 1);
% [~, d_idx_slice] = min(abs(d_values' - d_val_slice), [], 1);
% 
% % Parameter names
% param_names = {'d', '\theta', 'I'};
% 
% % Loop through each estimated parameter
% for param_idx = 1:3
%     figure;
%     set(gcf, 'Position', [100, 100, 1000, 600]); 
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
%         subplot(2, 3, subplot_idx+3);
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
%         xlim([0 90])
%         grid on;
%     end
% end
%% results in the paper
clear; close all; load("EdgeEstimation_results.mat");

theta_val_slice = [0.1, 0.3, 0.7];
d_val_slice = [0.05, 0.1, 0.2, 0.3];


% Find closest indices for theta_val_slice and d_val_slice
[~, theta_idx_slice] = min(abs(theta_values' - theta_val_slice), [], 1);
[~, d_idx_slice] = min(abs(d_values' - d_val_slice), [], 1);

% Parameter names
param_names = {'d', '\theta', 'I'};

% Loop through each estimated parameter
for param_idx = 2
    figure;
    set(gcf, 'Position', [100, 100, 1000, 280]); 
    % Plot sigma vs. d for fixed theta values
    % for subplot_idx = 1:3
    %     theta_idx = theta_idx_slice(subplot_idx);
    %     subplot(2, 3, subplot_idx);
    %     hold on;
    % 
    %     % Extract sigma values for different strategies
    %     sigma_fixed = squeeze(std_results_fixed(:, theta_idx, param_idx));
    %     sigma_adaptive = squeeze(std_results_adaptive(:, theta_idx, param_idx));
    %     sigma_intensity = squeeze(std_results_intensity(:, theta_idx, param_idx));
    % 
    %     % Plot sigma vs. d
    %     plot(d_values, sigma_intensity, '-b', 'LineWidth', 1.5, 'DisplayName', 'Intensity');hold on;
    %     plot(d_values, sigma_fixed, '-r', 'LineWidth', 1.5, 'DisplayName', 'Fixed');hold on;
    %     plot(d_values, sigma_adaptive, '-g', 'LineWidth', 1.5, 'DisplayName', 'Adaptive');
    % 
    % 
    %     xlabel('d values');
    %     ylabel(['\sigma_{', param_names{param_idx}, '}']);
    %     title(['\theta \approx ', num2str(rad2deg(theta_values(theta_idx)), '%.2f') char(176)]);
    %     legend show;
    %     grid on;
    % end

    % Plot sigma vs. theta for fixed d values
    for subplot_idx = 1:3
        d_idx = d_idx_slice(subplot_idx);
        subplot(1, 3, subplot_idx);
        hold on;

        % Extract sigma values for different strategies
        sigma_fixed = squeeze(std_results_fixed(d_idx, :, param_idx));
        sigma_adaptive = squeeze(std_results_adaptive(d_idx, :, param_idx));
        sigma_intensity = squeeze(std_results_intensity(d_idx, :, param_idx));

        % Plot sigma vs. theta
        plot(rad2deg(theta_values), sigma_intensity, '-b', 'LineWidth', 1.5, 'DisplayName', 'Intensity');hold on;
        plot(rad2deg(theta_values), sigma_fixed, '-r', 'LineWidth', 1.5, 'DisplayName', 'Fixed');hold on;
        plot(rad2deg(theta_values), sigma_adaptive, '-g', 'LineWidth', 1.5, 'DisplayName', 'Adaptive');
        

        xlabel('\theta values');
        ylabel(['\sigma_{', param_names{param_idx}, '}']);
        title(['d \approx ', num2str(d_values(d_idx), '%.2f')]);
        legend show;
        xlim([0 90])
        grid on;
    end
end
