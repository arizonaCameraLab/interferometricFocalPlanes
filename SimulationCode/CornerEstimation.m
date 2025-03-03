clear; clc; close all;
% Start parallel pool (if not already running)
if isempty(gcp('nocreate'))
    parpool;  % Adjust number of workers as needed
end

% Define measurement function to simulate corner detection
measure = @(x0, y0, theta, I, a, alpha1, alpha2) MeasureCorner(x0, y0, theta, I, a, alpha1, alpha2);

% Simulation Parameters
num_trials = 2000; % Number of Monte Carlo trials
total_measurements = 6; % Number of measurements per trial
total_photons = 1000000; % Total photons available for measurements
photons = total_photons / total_measurements; % Photons allocated per measurement
a_val = 0.5; % Sensor pair half distance for diagonal and anti-diagonal pair
I_true = 1;
% Define a function to add Poisson noise to measurement data
addPoisson = @(data, scale) poissrnd(data .* scale) ./ scale;

% Define x0, y0 range
x0_values = linspace(0, 0, 1);  % 5 points in range [0,1]
y0_values = linspace(-0.25, 0.25, 5);  % 5 points in range [0,1]

% Define theta_true range
theta_values = linspace(0.001, pi/2, 5);  % Varying theta

% Storage for standard deviation results (4 parameters: x0, y0, theta, I)
std_results_fixed = zeros(length(x0_values), length(y0_values), length(theta_values), 4);
std_results_adaptive = zeros(length(x0_values), length(y0_values), length(theta_values), 4);
std_results_intensity = zeros(length(x0_values), length(y0_values), length(theta_values), 4);
rand_ratio = 0.0;
% Define the 3-sigma filtering function
filter_std_3sigma = @(x) std(x( abs(x - mean(x)) < 3*std(x) ));

tic
for x0_idx = 1:length(x0_values)
    for y0_idx = 1:length(y0_values)
        x0_true = x0_values(x0_idx); % Set true distance
        y0_true = y0_values(y0_idx); % Set true angle
        
        % Parallelize the loop over theta_values
        for theta_idx = 1:length(theta_values)
            theta_true = theta_values(theta_idx);
            results_fixed = zeros(num_trials, 4);
            results_adaptive = zeros(num_trials, 4);
            results_intensity = zeros(num_trials, 4);

            parfor trial = 1:num_trials
                % ------ Intensity Measurement Strategy ------
                % Intensity measurments: meausre the intensity for each sensor
                % points
                % alpha1_intensity = zeros(total_measurements, 1);
                % alpha2_intensity = zeros(total_measurements, 1);
                G_intensity = measure(x0_true, y0_true, theta_true, I_true, a_val, 0, 0);
                G_meas_intensity = addPoisson(G_intensity, total_photons);
                obj_fun_intensity = @(params) sum(measure(params(1), params(2), params(3), params(4), a_val, 0, 0) ...
                                                    - G_meas_intensity .* log(measure(params(1), params(2), params(3), params(4), a_val, 0, 0)) );
                results_intensity(trial, :) = fminunc(obj_fun_intensity, [x0_true, y0_true, theta_true, I_true]+ rand_ratio * randn(1, 4));

                % ------ Fixed Measurement Strategy ------
                % Fixed measurment: Given operator H with angle:alpha to
                % operate on J:Mutual Intensity matrix 
                alpha_vals_fixed = linspace(-pi/2, pi/2, total_measurements); % Fixed measurement angles
                G_fixed = measure(x0_true, y0_true, theta_true, I_true, a_val, alpha_vals_fixed, alpha_vals_fixed); % Simulated measurement
                G_meas_fixed = addPoisson(G_fixed, photons);% Add Poisson noise

                % Define objective function for optimization
                obj_fun_fixed = @(params) sum(measure(params(1), params(2), params(3), params(4), a_val, alpha_vals_fixed, alpha_vals_fixed) ...
                                                - G_meas_fixed .* log(measure(params(1), params(2), params(3), params(4), a_val, alpha_vals_fixed, alpha_vals_fixed)) );
                % Optimize parameters using fminunc
                results_fixed(trial, :) = fminunc(obj_fun_fixed, [x0_true, y0_true, theta_true, I_true]+ rand_ratio * randn(1, 4));

                % ------ Adaptive Measurement Strategy ------
                % Adaptive measurments: diagonizing the measurments for
                % total_measurements steps to find the optimal alpha
                alpha1_adaptive = []; % potential adative steps for 1 pair sensor  
                alpha2_adaptive = [];
                G_meas_adaptive = [];
                params_est = [x0_true, y0_true, theta_true, I_true]; % Initial parameter estimate

                for step = 1:total_measurements
                    if step == 1 % First step is 0 to measure diagonal elements: Intensity
                        alpha1_next = 0; alpha2_next = 0;
                    else
                        % Find optimal angles maximizing measurement variation
                        % which would diagoinalize the matrix J
                        alpha_candidates = linspace(0, pi/2, 100);
                        [~, idx1] = max(arrayfun(@(a) range(measure(params_est(1), params_est(2), params_est(3), params_est(4), a_val, a, alpha2_adaptive)), alpha_candidates));
                        alpha1_next = alpha_candidates(idx1);
                        [~, idx2] = max(arrayfun(@(a) range(measure(params_est(1), params_est(2), params_est(3), params_est(4), a_val, alpha1_adaptive, a)), alpha_candidates));
                        alpha2_next = alpha_candidates(idx2);
                    end
                    % Update adaptive measurement angles alpha
                    alpha1_adaptive = [alpha1_adaptive; alpha1_next];
                    alpha2_adaptive = [alpha2_adaptive; alpha2_next];
                    new_measure = addPoisson(measure(x0_true, y0_true, theta_true, I_true, a_val, alpha1_next, alpha2_next), photons);

                    if step == 1
                        G_meas_adaptive = new_measure;
                    else
                        old_g = reshape(G_meas_adaptive, [], 4);
                        G_meas_adaptive = reshape([old_g; new_measure(:)'], [], 1);
                    end

                    % Optimize parameters using fminunc with random
                    % perturbation
                    obj_fun_adaptive = @(params) sum(measure(params(1), params(2), params(3), params(4), a_val, alpha1_adaptive, alpha2_adaptive) ...
                                                            - G_meas_adaptive .* log(measure(params(1), params(2), params(3), params(4), a_val, alpha1_adaptive, alpha2_adaptive)) );

                    params_est = fminunc(obj_fun_adaptive, [x0_true, y0_true, theta_true, I_true]+ rand_ratio * randn(1, 4));
                end
                results_adaptive(trial, :) = params_est;
            end

            for param_idx = 1:4
                std_results_fixed(x0_idx, y0_idx, theta_idx, param_idx) = std(results_fixed(:, param_idx));
                std_results_adaptive(x0_idx, y0_idx, theta_idx, param_idx) = std(results_adaptive(:, param_idx));
                std_results_intensity(x0_idx, y0_idx, theta_idx, param_idx) = std(results_intensity(:, param_idx));
            end
        end
    end
end
toc

%%
% Save all results in one file
file_name = "CornerEstimation_results";
save(file_name);
%%
% close all;clc;load("CornerEstimation_results")
% % 
% theta_val_slice = [0.1, 1, 1.5];  
% y0_slice = [-0.2, 0.0, 0.2];
% 
% % find index of selected slice
% [~, theta_idx_slice] = arrayfun(@(x) min(abs(theta_values - x)), theta_val_slice);
% [~, y0_idx_slice] = arrayfun(@(y) min(abs(y0_values - y)), y0_slice);
% 
% % parameter label
% param_labels = {'x_0', 'y_0', '\theta', 'I'};
% 
% % loop all paramters
% for param_idx = 1:3
%     figure;
%     set(gcf, 'Position', [100, 100, 1000, 600]); 
%     % sigma vs. y0, fixed theta
%     for i = 1:length(theta_idx_slice)
%         subplot(2,3,i);
%         theta_idx = theta_idx_slice(i);
% 
%         % extract data
%         sigma_fixed = squeeze(std_results_fixed(1,:,theta_idx,param_idx));
%         sigma_intensity = squeeze(std_results_intensity(1,:,theta_idx,param_idx));
%         sigma_adaptive = squeeze(std_results_adaptive(1,:,theta_idx,param_idx));
% 
%         % plot
%         plot(y0_values, sigma_intensity, '-b', 'LineWidth', 1.5, 'DisplayName', 'Intensity');hold on;
%         plot(y0_values, sigma_fixed, '-r', 'LineWidth', 1.5, 'DisplayName', 'Fixed'); hold on;
% 
%         plot(y0_values, sigma_adaptive, '-g', 'LineWidth', 1.5, 'DisplayName', 'Adaptive');
% 
%         xlabel('y_0'); ylabel(['\sigma_{', param_labels{param_idx}, '}']);
%         title(['\theta = ', num2str(rad2deg(theta_values(theta_idx)), '%.2f') char(176)]);
%         legend show;
%         grid on;
%     end
% 
%     % : sigma vs. theta, fix y0
%     for i = 1:length(y0_idx_slice)
%         subplot(2,3,i+3);
%         y0_idx = y0_idx_slice(i);
% 
%         % extract data
%         sigma_fixed = squeeze(std_results_fixed(1,y0_idx,:,param_idx));
%         sigma_intensity = squeeze(std_results_intensity(1,y0_idx,:,param_idx));
%         sigma_adaptive = squeeze(std_results_adaptive(1,y0_idx,:,param_idx));
% 
%         % plot
%         plot(rad2deg(theta_values), sigma_intensity, '-b', 'LineWidth', 1.5, 'DisplayName', 'Intensity');hold on;
%         plot(rad2deg(theta_values), sigma_fixed, '-r', 'LineWidth', 1.5, 'DisplayName', 'Fixed'); hold on;
% 
%         plot(rad2deg(theta_values), sigma_adaptive, '-g', 'LineWidth', 1.5, 'DisplayName', 'Adaptive');
% 
%         xlabel(['\theta' ' (' char(176) ')']); ylabel(['\sigma_{', param_labels{param_idx}, '}']);
%         title(['y_0 = ', num2str(y0_values(y0_idx), '%.2f')]);
%         legend show;
%         xlim([0 90])
%         grid on;
%     end
% 
% end
%% plot results in paper
close all;clc;load("CornerEstimation_results")
% 
theta_val_slice = [0.1, 1, 1.5];  
y0_slice = [-0.2, 0.0, 0.2];

% find index of selected slice
[~, theta_idx_slice] = arrayfun(@(x) min(abs(theta_values - x)), theta_val_slice);
[~, y0_idx_slice] = arrayfun(@(y) min(abs(y0_values - y)), y0_slice);

% data label
param_labels = {'x_0', 'y_0', '\theta', 'I'};

% loop 2 parameters
for param_idx = 3
    figure;
    set(gcf, 'Position', [100, 100, 1000, 280]); 
    % sigma vs. y0, fix theta
    % for i = 1:length(theta_idx_slice)
    %     subplot(2,3,i);
    %     theta_idx = theta_idx_slice(i);
    % 
    %     % extract data
    %     sigma_fixed = squeeze(std_results_fixed(1,:,theta_idx,param_idx));
    %     sigma_intensity = squeeze(std_results_intensity(1,:,theta_idx,param_idx));
    %     sigma_adaptive = squeeze(std_results_adaptive(1,:,theta_idx,param_idx));
    % 
    %     % plot
    %     plot(y0_values, sigma_intensity, '-b', 'LineWidth', 1.5, 'DisplayName', 'Intensity');hold on;
    %     plot(y0_values, sigma_fixed, '-r', 'LineWidth', 1.5, 'DisplayName', 'Fixed'); hold on;
    % 
    %     plot(y0_values, sigma_adaptive, '-g', 'LineWidth', 1.5, 'DisplayName', 'Adaptive');
    % 
    %     xlabel('y_0'); ylabel(['\sigma_{', param_labels{param_idx}, '}']);
    %     title(['\theta = ', num2str(rad2deg(theta_values(theta_idx)), '%.2f') char(176)]);
    %     legend show;
    %     grid on;
    % end

    % sigma vs. theta, fix y0
    for i = 1:length(y0_idx_slice)
        subplot(1,3,i+3-3);
        y0_idx = y0_idx_slice(i);

        % extract data
        sigma_fixed = squeeze(std_results_fixed(1,y0_idx,:,param_idx));
        sigma_intensity = squeeze(std_results_intensity(1,y0_idx,:,param_idx));
        sigma_adaptive = squeeze(std_results_adaptive(1,y0_idx,:,param_idx));

        % plot
        plot(rad2deg(theta_values), sigma_intensity, '-b', 'LineWidth', 1.5, 'DisplayName', 'Intensity');hold on;
        plot(rad2deg(theta_values), sigma_fixed, '-r', 'LineWidth', 1.5, 'DisplayName', 'Fixed'); hold on;

        plot(rad2deg(theta_values), sigma_adaptive, '-g', 'LineWidth', 1.5, 'DisplayName', 'Adaptive');

        xlabel(['\theta' ' (' char(176) ')'], 'FontSize', 15); 
        ylabel(['Estimation Error of {', param_labels{param_idx}, '}'], 'FontSize', 15);
        % title(['y_0 = ', num2str(y0_values(y0_idx), '%.2f')], 'FontSize', 17);
        xlim([0 90])
        if i == 1
            lgd = legend('show');
            set(lgd, 'FontSize', 12); % Set legend font size
        end
        grid on;
    end

end
