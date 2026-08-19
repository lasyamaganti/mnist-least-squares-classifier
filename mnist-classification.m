clear;
close all;

num_train = 5000;
[train_imgs, train_labels] = readMNIST('train-images-idx3-ubyte', 'train-labels-idx1-ubyte', num_train, 0);

threshold = 600; % # of imgs that have non-zero intensity for pixel

% train-images is a 20x20x5000 3d array
nonzero_counts = sum(train_imgs ~= 0, 3); % creates a logical array the same size as train_imgs where every image that is nonzero becomes "true" (1) and every pixel that is exactly zero becomes "false" (0). then sums along the 3rd index, adding up all the 1/ nonzero pixels acros the 5000 images.

% nonzero_counts is a 20x20 matrix where each entry has an integer between
% 0 and 5000 for the number of pixels where the intensity is nonzero

feature_mask = nonzero_counts >= threshold; 

feature_idx = find(feature_mask); % M0x1, where M0 is #features; gets indices of pixels that exceed threshold to pull them put as features
M0 = numel(feature_idx);
fprintf('Number of features with >=600 nonzeros: %d\n', M0);

%% Building Matrix A and vector y

N = num_train;

A = zeros(N, M0);
y = zeros(N, 1);

% Each image has a vector length of M0 (for the pixels that exceed the
% threshold.
for k = 1:N % loop through the images and build one row of A per image
    imgk = train_imgs(:,:,k); % take k-th image; imgk is a 20x20 matrix with the pixel intensities of that image
    img_vec = imgk(:); % vectorizes the image, so stacks the 2d matrix into a 400ox1 column vector
    A(k,:) = img_vec(feature_idx); % 1xM0; takes the pixels exceeding the threshold that were indexed previously from the column vector for the image which gives a row vector of length M0 which is assigned the the k row of matrix A.
    
    % from the project 2 description, focus on determining whether the
    % image is 0 corresponding to class 1 or digits 1-9 corresponding to
    % class -1
    if train_labels(k) == 0
        y(k) = 1;
    else
        y(k) = -1;
    end
end

theta = A \ y; % Matlab back slash is least square

% theta is the length of M0, but it needs to be put back in to a 20x20
% image at the featured pixels.
theta_img = zeros(20, 20);
theta_img(feature_idx) = theta;

figure;
imagesc(theta_img);
colorbar;
title('\theta values at pixel locations');
axis image;

%% Use training to test on new set

num_test = 5000;
[test_imgs, test_labels] = readMNIST('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte', num_test, 0);

y_true = zeros(num_test, 1); % true class converted into -1 or 1 to match the convention.
y_hat  = zeros(num_test, 1); % predicted label (-1 or 1) for test image

% do same thing as the training set to get the image in column vector  and
% use the same feature index as the training to create an M0x1 vector with
% the image information about pixels that exceed the threshold.
for k = 1:num_test
    imgk = test_imgs(:, :, k);
    img_vec = imgk(:);
    xk = img_vec(feature_idx); % M0x1

    score = theta' * xk; % scalar; linear combination of features; how strong the image aligns with theta
    if score > 0 % if score is positive, classifier leans toward +1 or digit 0
        y_hat(k) = 1;
    elseif score < 0 % if score is negative, classifier leans toward class -1 or digits 1-9
        y_hat(k) = -1;
    else
        y_hat(k) = 0;  % sign(0) - rarely happens
    end

    if test_labels(k) == 0
        y_true(k) = 1;
    else
        y_true(k) = -1;
    end
end

% overall error
num_err = sum(y_hat ~= y_true); % # of test image classified incorrectly
err_rate = num_err / num_test; % error rate = # of test images classified incorrectly / total test images

% false positive: predicted 1, true -1
is_neg = (y_true == -1); % is digits 1-9
fp = sum(y_hat == 1 & is_neg); % predicted  digit 0, actually digits 1-9
fp_rate = fp / sum(is_neg); % false pos / actually -1

% false negative: predicted -1, true 1
is_pos = (y_true == 1); % is digit 0
fn = sum(y_hat == -1 & is_pos); % predicted digits 1-9, actually digit 0
fn_rate = fn / sum(is_pos); % false neg / actually 1

fprintf('Full set (5000) -> test results:\n');
fprintf('Error rate: %.4f\n', err_rate);
fprintf('False positive rate: %.4f\n', fp_rate);
fprintf('False negative rate: %.4f\n', fn_rate);

%% Reduced training set

% do not need to recalculate M0 because same amount of pixels were nonzero
% in >600 of the original 500 images

Nsmall = 100; % smaller set to train classifier
A_small = zeros(Nsmall, M0);
y_small = zeros(Nsmall, 1);

for k = 1:Nsmall
    imgk = train_imgs(:, :, k);
    img_vec = imgk(:);
    A_small(k, :) = img_vec(feature_idx);

    if train_labels(k) == 0
        y_small(k) = 1;
    else
        y_small(k) = -1;
    end
end

theta_small = A_small \ y_small;

%% Use reduced training set to test on new set

y_hat_small = zeros(num_test,1);

for k = 1:num_test
    imgk = test_imgs(:, :, k);
    img_vec = imgk(:);
    xk = img_vec(feature_idx);

    score = theta_small' * xk;
    if score > 0
        y_hat_small(k) = 1;
    elseif score < 0
        y_hat_small(k) = -1;
    else
        y_hat_small(k) = 0;
    end
end

% total error
num_err = sum(y_hat_small ~= y_true);
err_rate_small = num_err / num_test;

% false positive
fp = sum(y_hat_small == 1 & is_neg);
fp_rate_small = fp / sum(is_neg);

% false negative
fn = sum(y_hat_small == -1 & is_pos);
fn_rate_small = fn / sum(is_pos);

fprintf('\nSmall training set (100) -> test results:\n');
fprintf('Error rate: %.4f\n', err_rate_small);
fprintf('False positive rate: %.4f\n', fp_rate_small);
fprintf('False negative rate: %.4f\n', fn_rate_small);

%% Changing the feature set

M_list = [20, 50, 1000, 5000, 10000];

for mm = 1:length(M_list)
    M = M_list(mm);
    fprintf('\n Random feature size M = %d \n', M);

    % random 1 or -1 matrix
    R = sign(randn(M, M0)); % M x M0

    % build new training feature matrix still using 5000 training samples
    A_rand = zeros(N, M);
    for k = 1:N
        imgk = train_imgs(:, :, k);
        x_old = imgk(:);
        x_old = x_old(feature_idx); % M0x1
        x_new = R * x_old; % Mx1
        x_new = max(x_new, 0); 
        A_rand(k, :) = x_new.';
    end

    % solve least squares with same y
    theta_rand = A_rand \ y;

    % test
    y_hat_rand = zeros(num_test, 1);
    for k = 1:num_test
        imgk = test_imgs(:, :, k);
        x_old = imgk(:);
        x_old = x_old(feature_idx);
        x_new = R * x_old;
        x_new = max(x_new, 0);

        score = theta_rand' * x_new;
        if score > 0
            y_hat_rand(k) = 1;
        elseif score < 0
            y_hat_rand(k) = -1;
        else
            y_hat_rand(k) = 0;
        end
    end
    
    % total error
    num_err = sum(y_hat_rand ~= y_true);
    err_rate_rand = num_err / num_test;

    % false positive
    fp = sum(y_hat_rand == 1 & is_neg);
    fp_rate_rand = fp / sum(is_neg);
    
    % false negative
    fn = sum(y_hat_rand == -1 & is_pos);
    fn_rate_rand = fn / sum(is_pos);

    fprintf('Error rate: %.4f\n', err_rate_rand);
    fprintf('False positive rate: %.4f\n', fp_rate_rand);
    fprintf('False negative rate: %.4f\n', fn_rate_rand);
end
