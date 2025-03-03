clear; clc; close all;

% forward model
measure = @(x1, y1, x2, y2, I1, I2, a, alpha1, alpha2) MeasureI2P2D(x1, y1, x2, y2, I1, I2, a, alpha1, alpha2);
addPoisson = @(data, scale) poissrnd(data .* scale) ./ scale;
truncnorm = @(lb, ub, x, shape) max(min(x*randn(shape), ub), lb);

% settings
num_trials = 2000;
total_measurements = 6;
total_photons = 100000;
photons = total_photons / total_measurements;
a_val = 0.5;
I1_true = 1.0;
I2_true = 10;

% parameters scanning range
x1_values = linspace(0, 1, 11);
y1_values = linspace(0.1, 0.5, 5);

% results: Fixed, Adaptive, Intensity
std_x1_fixed = zeros(length(x1_values), length(y1_values));
std_y1_fixed = zeros(length(x1_values), length(y1_values));
std_x2_fixed = zeros(length(x1_values), length(y1_values));
std_y2_fixed = zeros(length(x1_values), length(y1_values));
std_dist_fixed = zeros(length(x1_values), length(y1_values));

std_x1_adaptive = zeros(length(x1_values), length(y1_values));
std_y1_adaptive = zeros(length(x1_values), length(y1_values));
std_x2_adaptive = zeros(length(x1_values), length(y1_values));
std_y2_adaptive = zeros(length(x1_values), length(y1_values));
std_dist_adaptive = zeros(length(x1_values), length(y1_values));

std_x1_intensity = zeros(length(x1_values), length(y1_values));
std_y1_intensity = zeros(length(x1_values), length(y1_values));
std_x2_intensity = zeros(length(x1_values), length(y1_values));
std_y2_intensity = zeros(length(x1_values), length(y1_values));
std_dist_intensity = zeros(length(x1_values), length(y1_values));

std_mean_x_fixed = zeros(length(x1_values), length(y1_values));
std_mean_y_fixed = zeros(length(x1_values), length(y1_values));
std_diff_x_fixed = zeros(length(x1_values), length(y1_values));
std_diff_y_fixed = zeros(length(x1_values), length(y1_values));

std_mean_x_adaptive = zeros(length(x1_values), length(y1_values));
std_mean_y_adaptive = zeros(length(x1_values), length(y1_values));
std_diff_x_adaptive = zeros(length(x1_values), length(y1_values));
std_diff_y_adaptive = zeros(length(x1_values), length(y1_values));

std_mean_x_intensity = zeros(length(x1_values), length(y1_values));
std_mean_y_intensity = zeros(length(x1_values), length(y1_values));
std_diff_x_intensity = zeros(length(x1_values), length(y1_values));
std_diff_y_intensity = zeros(length(x1_values), length(y1_values));

% parameters scan
for y_idx = 1:length(y1_values)
    for x1_idx = 1:length(x1_values)
        x1_true = x1_values(x1_idx);
        x2_true = x1_true;  % assume x1 = x2
        y1_true = y1_values(y_idx);
        y2_true = -y1_true; % assume y1 = -y2

        results_fixed = zeros(num_trials, 4); 
        results_adaptive = zeros(num_trials, 4);
        results_intensity = zeros(num_trials, 4);

        parfor trial = 1:num_trials
            %% ========== Fixed Strategy ==========
            alpha1_fixed = linspace(-pi/2, pi/2, total_measurements);
            alpha2_fixed = linspace(-pi/2, pi/2, total_measurements);
            G_fixed = measure(x1_true, y1_true, x2_true, y2_true, I1_true, I2_true, a_val, alpha1_fixed, alpha2_fixed);
            G_meas_fixed = addPoisson(G_fixed, photons);
            obj_fun_fixed = @(params) -sum(G_meas_fixed .* log(measure(params(1), params(2), params(3), params(4), ...
                                I1_true, I2_true, a_val, alpha1_fixed, alpha2_fixed)) ...
                                - measure(params(1), params(2), params(3), params(4), ...
                                I1_true, I2_true, a_val, alpha1_fixed, alpha2_fixed));
            results_fixed(trial, :) = fminunc(obj_fun_fixed, [x1_true, y1_true, x2_true, y2_true] + ...
                                        truncnorm(-0.05, 0.05, 0.05, [1,4]));

            %% ========== Adaptive Strategy ==========
            alpha1_adaptive = [];
            alpha2_adaptive = [];
            G_meas_adaptive = [];
            params_est = [x1_true, y1_true, x2_true, y2_true];
            for step = 1:total_measurements
                if step == 1
                    alpha1_next = 0; 
                    alpha2_next = 0;
                else
                    alpha_candidates = linspace(0, pi/2, 100);
                    [~, idx1] = max(arrayfun(@(a) range(measure(params_est(1), params_est(2), params_est(3), params_est(4), ...
                                        I1_true, I2_true, a_val, a, alpha2_adaptive)), alpha_candidates));
                    alpha1_next = alpha_candidates(idx1);
                    [~, idx2] = max(arrayfun(@(a) range(measure(params_est(1), params_est(2), params_est(3), params_est(4), ...
                                        I1_true, I2_true, a_val, alpha1_adaptive, a)), alpha_candidates));
                    alpha2_next = alpha_candidates(idx2);
                end
                alpha1_adaptive = [alpha1_adaptive; alpha1_next];
                alpha2_adaptive = [alpha2_adaptive; alpha2_next];
                new_measure = addPoisson(measure(x1_true, y1_true, x2_true, y2_true, I1_true, I2_true, a_val, alpha1_next, alpha2_next), photons);
                if step == 1
                    G_meas_adaptive = new_measure;
                else
                    old_g = reshape(G_meas_adaptive, [], 4);
                    G_meas_adaptive = reshape([old_g; new_measure(:)'], [], 1);
                end
                obj_fun_adaptive = @(params) -sum(G_meas_adaptive .* log(measure(params(1), params(2), params(3), params(4), ...
                                    I1_true, I2_true, a_val, alpha1_adaptive, alpha2_adaptive)) ...
                                    - measure(params(1), params(2), params(3), params(4), ...
                                    I1_true, I2_true, a_val, alpha1_adaptive, alpha2_adaptive));
                params_est = fminunc(obj_fun_adaptive, [x1_true, y1_true, x2_true, y2_true] + ...
                                    truncnorm(-0.05, 0.05, 0.05, [1,4]));
            end
            results_adaptive(trial, :) = params_est;
            
            %% ========== Intensity Strategy ==========
            % alpha=0
            alpha1_intensity = zeros(total_measurements, 1);
            alpha2_intensity = zeros(total_measurements, 1);
            G_intensity = measure(x1_true, y1_true, x2_true, y2_true, I1_true, I2_true, a_val, alpha1_intensity, alpha2_intensity);
            G_meas_intensity = addPoisson(G_intensity, photons);
            obj_fun_intensity = @(params) -sum(G_meas_intensity .* log(measure(params(1), params(2), params(3), params(4), ...
                                    I1_true, I2_true, a_val, alpha1_intensity, alpha2_intensity)) ...
                                    - measure(params(1), params(2), params(3), params(4), ...
                                    I1_true, I2_true, a_val, alpha1_intensity, alpha2_intensity));
            results_intensity(trial, :) = fminunc(obj_fun_intensity, [x1_true, y1_true, x2_true, y2_true] + ...
                                        truncnorm(-0.05, 0.05, 0.05, [1,4]));
        end

        % calculate std. May use other functions to filter outlier, here we
        % directly use std without any filtering
        filter_std = @(x) std(x);

        %% Fixed Strategy
        std_x1_fixed(x1_idx, y_idx) = filter_std(results_fixed(:,1));
        std_y1_fixed(x1_idx, y_idx) = filter_std(results_fixed(:,2));
        std_x2_fixed(x1_idx, y_idx) = filter_std(results_fixed(:,3));
        std_y2_fixed(x1_idx, y_idx) = filter_std(results_fixed(:,4));
        std_dist_fixed(x1_idx, y_idx) = filter_std(sqrt((results_fixed(:,1) - results_fixed(:,3)).^2 + ...
                                                      (results_fixed(:,2) - results_fixed(:,4)).^2));
        std_mean_x_fixed(x1_idx, y_idx) = filter_std((results_fixed(:,1) + results_fixed(:,3)) / 2);
        std_mean_y_fixed(x1_idx, y_idx) = filter_std((results_fixed(:,2) + results_fixed(:,4)) / 2);
        std_diff_x_fixed(x1_idx, y_idx) = filter_std(abs(results_fixed(:,1) - results_fixed(:,3)));
        std_diff_y_fixed(x1_idx, y_idx) = filter_std(abs(results_fixed(:,2) - results_fixed(:,4)));

        %% Adaptive Strategy
        std_x1_adaptive(x1_idx, y_idx) = filter_std(results_adaptive(:,1));
        std_y1_adaptive(x1_idx, y_idx) = filter_std(results_adaptive(:,2));
        std_x2_adaptive(x1_idx, y_idx) = filter_std(results_adaptive(:,3));
        std_y2_adaptive(x1_idx, y_idx) = filter_std(results_adaptive(:,4));
        std_dist_adaptive(x1_idx, y_idx) = filter_std(sqrt((results_adaptive(:,1) - results_adaptive(:,3)).^2 + ...
                                                         (results_adaptive(:,2) - results_adaptive(:,4)).^2));
        std_mean_x_adaptive(x1_idx, y_idx) = filter_std((results_adaptive(:,1) + results_adaptive(:,3)) / 2);
        std_mean_y_adaptive(x1_idx, y_idx) = filter_std((results_adaptive(:,2) + results_adaptive(:,4)) / 2);
        std_diff_x_adaptive(x1_idx, y_idx) = filter_std(abs(results_adaptive(:,1) - results_adaptive(:,3)));
        std_diff_y_adaptive(x1_idx, y_idx) = filter_std(abs(results_adaptive(:,2) - results_adaptive(:,4)));

        %% Intensity Strategy
        std_x1_intensity(x1_idx, y_idx) = filter_std(results_intensity(:,1));
        std_y1_intensity(x1_idx, y_idx) = filter_std(results_intensity(:,2));
        std_x2_intensity(x1_idx, y_idx) = filter_std(results_intensity(:,3));
        std_y2_intensity(x1_idx, y_idx) = filter_std(results_intensity(:,4));
        std_dist_intensity(x1_idx, y_idx) = filter_std(sqrt((results_intensity(:,1) - results_intensity(:,3)).^2 + ...
                                                           (results_intensity(:,2) - results_intensity(:,4)).^2));
        std_mean_x_intensity(x1_idx, y_idx) = filter_std((results_intensity(:,1) + results_intensity(:,3)) / 2);
        std_mean_y_intensity(x1_idx, y_idx) = filter_std((results_intensity(:,2) + results_intensity(:,4)) / 2);
        std_diff_x_intensity(x1_idx, y_idx) = filter_std(abs(results_intensity(:,1) - results_intensity(:,3)));
        std_diff_y_intensity(x1_idx, y_idx) = filter_std(abs(results_intensity(:,2) - results_intensity(:,4)));
    end
end

% save results for future analysus
save('TwoPointsMeasurementKnownIntensity.mat', 'x1_values', 'y1_values', ...
     'std_x1_fixed', 'std_y1_fixed', 'std_x2_fixed', 'std_y2_fixed', 'std_dist_fixed', ...
     'std_mean_x_fixed', 'std_mean_y_fixed', 'std_diff_x_fixed', 'std_diff_y_fixed', ...
     'std_x1_adaptive', 'std_y1_adaptive', 'std_x2_adaptive', 'std_y2_adaptive', 'std_dist_adaptive', ...
     'std_mean_x_adaptive', 'std_mean_y_adaptive', 'std_diff_x_adaptive', 'std_diff_y_adaptive', ...
     'std_x1_intensity', 'std_y1_intensity', 'std_x2_intensity', 'std_y2_intensity', 'std_dist_intensity', ...
     'std_mean_x_intensity', 'std_mean_y_intensity', 'std_diff_x_intensity', 'std_diff_y_intensity');

disp('Data saved successfully!');

%% Plot figures
close all
selected_y_indices = [1,3,5];

% Example 1: Distance
plotThreeSubplots(x1_values, std_dist_fixed, std_dist_adaptive, std_dist_intensity, ...
                  selected_y_indices, y1_values, '\Deltax');

% Example 2: Mean X
plotThreeSubplots(x1_values, std_mean_x_fixed, std_mean_x_adaptive, std_mean_x_intensity, ...
                  selected_y_indices, y1_values, '(x1+x2)/2');

% Example 3: Mean Y
plotThreeSubplots(x1_values, std_mean_y_fixed, std_mean_y_adaptive, std_mean_y_intensity, ...
                  selected_y_indices, y1_values, '(y1+y2)/2');

% Example 4: |x1 - x2|
plotThreeSubplots(x1_values, std_diff_x_fixed, std_diff_x_adaptive, std_diff_x_intensity, ...
                  selected_y_indices, y1_values, '|x1-x2|');

% Example 5: |y1 - y2|
plotThreeSubplots(x1_values, std_diff_y_fixed, std_diff_y_adaptive, std_diff_y_intensity, ...
                  selected_y_indices, y1_values, '|y1-y2|');

function plotThreeSubplots(x_vals, dataF, dataA, dataI, y_indices, y_vals, statName)
%  - x_vals  : x1_values
%  - dataF   : Fixed strategy [N x M]
%  - dataA   : Adaptive strategy [N x M]
%  - dataI   : Intensity strategy [N x M]
%  - y_indices : select some y to plor, eg, [1,3,5]
%  - y_vals    : y1_values vector
%  - statName  : Distance,(x1+x2)/2, etc

    colors = {'b', 'r', 'g'}; % Fixed = red, Adaptive = blue, Intensity = green
    strategies = {'Fixed', 'Adaptive', 'Intensity'};
    
    figure;
    
    for sub_i = 1:length(y_indices)
        y_idx = y_indices(sub_i);
        subplot(1, length(y_indices), sub_i); hold on;
        data_I_crop = dataI(:, y_idx);
        data_F_crop = dataF(:, y_idx);
        data_A_crop = dataA(:, y_idx);
        %  Intensity
        plot(x_vals(1:end/2+1), data_I_crop(1:end/2+1), '-', 'Color', colors{1}, 'LineWidth', 1.5, ...
             'DisplayName', strategies{3});
        % Fixed
        plot(x_vals(1:end/2+1), data_F_crop(1:end/2+1), '-', 'Color', colors{2}, 'LineWidth', 1.5, ...
             'DisplayName', strategies{1});
        
        % Adaptive
        plot(x_vals(1:end/2+1), data_A_crop(1:end/2+1), '-', 'Color', colors{3}, 'LineWidth', 1.5, ...
             'DisplayName', strategies{2});
        

        
        xlabel('x_1', 'FontSize', 15);
        ylabel(sprintf('Estimation Error of {%s}', statName), 'FontSize', 15);
        title(sprintf('%s =%.2f', statName, y_vals(y_idx)*2), 'FontSize', 17);
        legend('show');
        grid on;
        xlim([0 0.5])
        set(gca, 'FontSize', 11)
        if sub_i == 1
            lgd = legend('show');
            set(lgd, 'FontSize', 12); % Set legend font size
        end
    end
    
    set(gcf, 'Position', [100, 100, 1000, 280]); 
end



