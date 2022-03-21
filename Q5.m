% Q5
clc;clear;
% a)
full_image = zeros(2500,240);

file_name=dir(strcat('/Users/shiqi/Downloads/271b/hw/trainset')); 
for i=4:length(file_name) %suppose there are 10 image
% the path tht u hv imges
  im = im2double(imread(strcat('/Users/shiqi/Downloads/271b/hw/trainset/',file_name(i).name)));
  im_r = reshape(im, [2500,1]);
  c=i-3;
  full_image(:,c) = im_r;
end

mean_f = mean(full_image);
center_f = full_image - repmat(mean_f,[2500,1]);
[U S V] = svd(center_f);
basis = center_f * V(:,1:16);
pic = reshape(basis(:,1:16),[50,50,16]);
for i = 1:16
    subplot(4,4,i);
    imshow(pic(:,:,i),[]);
    axis on 
end

%b)
P1 = full_image(:,1:40);
P2 = full_image(:,41:80);
P3 = full_image(:,81:120);
P4 = full_image(:,121:160);
P5 = full_image(:,161:200);
P6 = full_image(:,201:240);

full_P = zeros(2500,40,6);
full_P(:,:,1) = P1;
full_P(:,:,2) = P2;
full_P(:,:,3) = P3;
full_P(:,:,4) = P4;
full_P(:,:,5) = P5;
full_P(:,:,6) = P6;

i = 1
j = 2
r = 0

W = zeros(2500,15);
for i = 1:5
    for j = (i+1):6
        Px = full_P(:,:,i);
        Py = full_P(:,:,j);
        w_xy = inv(cov(Px')+cov(Py')+eye(2500)) * (mean(Px,2)-mean(Py,2));
        r = r + 1
        W(:,r) = w_xy;
    end
end

W(:,16) = zeros(2500,1);
pic = reshape(W(:,1:16),[50,50,16]);
for i = 1:16
    subplot(4,4,i);
    imshow(pic(:,:,i),[]);
    axis on 
end

% c)
basis_15 = center_f * V(:,1:15);
phi = basis_15';

mu = zeros(15,6);
sigma = zeros(15,15,6);
for i = 1:6
    Z = phi * full_P(:,:,i);
    mu(:,i) = mean(Z,2);
    sigma(:,:,i) = cov(Z');
end

test_image = zeros(2500,60);

file_name=dir(strcat('/Users/shiqi/Downloads/271b/hw/testset')); 
for i=4:length(file_name) %suppose there are 10 image
% the path tht u hv imges
  im = im2double(imread(strcat('/Users/shiqi/Downloads/271b/hw/testset/',file_name(i).name)));
  im_r = reshape(im, [2500,1]);
  c=i-3;
  test_image(:,c) = im_r;
end


P1_t = test_image(:,1:10);
P2_t = test_image(:,11:20);
P3_t = test_image(:,21:30);
P4_t = test_image(:,31:40);
P5_t = test_image(:,41:50);
P6_t = test_image(:,51:60);

full_t = zeros(2500,10,6);
full_t(:,:,1) = P1_t;
full_t(:,:,2) = P2_t;
full_t(:,:,3) = P3_t;
full_t(:,:,4) = P4_t;
full_t(:,:,5) = P5_t;
full_t(:,:,6) = P6_t;



Z_t = zeros(15,10,6);
for i = 1:6
    Z_t(:,:,i) = phi * full_t(:,:,i);
end

% put the test 1 into train 1 and train 2
prior = 1/2
test_error = zeros(6,1);

for i = 1:6
    test_result = [];
    for j = 1:6
        if i ~= j
            testi = log_pdf(Z_t(:,:,i),mu(:,i),sigma(:,:,i)) > log_pdf(Z_t(:,:,i),mu(:,j),sigma(:,:,j));
            test_result = [test_result;testi];
        end
    end
    test_error(i) = 1-sum(test_result(:))/numel(test_result);
end

test_error_all = mean(test_error);

% d)
W_test = W(:,1:15);

test_error = zeros(6,1);

for i = 1:6
    test_result = [];
    for j = 1:6
        if i ~= j
            Px = full_P(:,:,i);
            Py = full_P(:,:,j);
            w_xy = inv(cov(Px')+cov(Py')+eye(2500)) * (mean(Px,2)-mean(Py,2));
            
            Zi = w_xy' * full_P(:,:,i);
            mui = mean(Zi);
            sigmai = cov(Zi');
            
            Zj = w_xy' * full_P(:,:,j);
            muj = mean(Zj);
            sigmaj = cov(Zj');
            
            Z_t = w_xy' * full_t(:,:,i);
            
            testi = log_pdf(Z_t,mui,sigmai) > log_pdf(Z_t,muj,sigmaj);
            
            test_result = [test_result;testi];
        end
    end
    test_error(i) = 1-sum(test_result(:))/numel(test_result);
end
test_error_all = mean(test_error);

% e)

basis_30 = center_f * V(:,1:30);
phi_30 = basis_30';

for i = 1:6
    test_result = [];
    for j = 1:6
        if i ~= j
            Px = phi_30 * full_P(:,:,i);
            Py = phi_30 * full_P(:,:,j);
            w_xy = inv(cov(Px')+cov(Py')) * (mean(Px,2)-mean(Py,2));
            
            Zi = w_xy' * Px;
            mui = mean(Zi);
            sigmai = cov(Zi');
            
            Zj = w_xy' * Py;
            muj = mean(Zj);
            sigmaj = cov(Zj');
            
            Z_t = w_xy' * (phi_30 * full_t(:,:,i));
            
            testi = log_pdf(Z_t,mui,sigmai) > log_pdf(Z_t,muj,sigmaj);
            
            test_result = [test_result;testi];
        end
    end
    test_error(i) = 1-sum(test_result(:))/numel(test_result);
end



