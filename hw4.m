%% load data
clc;
clear all;

[train_imgs, train_labels] = readMNIST('train-images.idx3-ubyte','train-labels.idx1-ubyte', 20000, 0);
assert(all(size(train_imgs)==[20000,784]));
assert(all(size(train_labels)==[20000,1]));

[test_imgs, test_labels] = readMNIST('t10k-images.idx3-ubyte','t10k-labels.idx1-ubyte', 10000, 0);
assert(all(size(test_imgs)==[10000,784]));
assert(all(size(test_labels)==[10000,1]));


%% (a) linear svm

c_total = [2,4,8];
c = c_total(3);
test_accuracy_c = zeros(10,1);
num_sv_c = zeros(10,1);
maxPosAlphaIndex_c = zeros(10,3);
maxNegAlphaIndex_c = zeros(10,3);
marginTrain = zeros(10,20000);
marginTest = zeros(10,10000);


addpath /Users/shiqi/Downloads/271b/hw4/libsvm-3.24/matlab

%%
for class_i = 0:9
    train_labels_i = -ones(size(train_labels));
    train_labels_i(train_labels==class_i) = 1;
    test_labels_i = -ones(size(test_labels));
    test_labels_i(test_labels==class_i) = 1;
    % libsvm package
    % c: regularization constant
    % t: kernel_type
    % -t 1 linear: u'*v
    config = sprintf('-c %d -t %d', c, 0);
    model_svm_linear = svmtrain(train_labels_i, train_imgs, config);
    [predict_label, accuracy, dec_values] = svmpredict(test_labels_i, test_imgs, model_svm_linear);
    test_accuracy_c(class_i+1) = accuracy(1);
    num_sv_c(class_i+1) = model_svm_linear.('totalSV');
    nSV = model_svm_linear.('nSV');
    pos = nSV(1);
    [~,maxPosIndex] = maxk(model_svm_linear.('sv_coef')(1:pos),3);
    [~,maxNegIndex] = maxk(model_svm_linear.('sv_coef')(pos+1:end),3);
    
    [~,sortPosIndex] = sort(-model_svm_linear.('sv_coef')(1:pos));
    [~,sortNegIndex] = sort(model_svm_linear.('sv_coef')(pos+1:end));
    maxPosIndex = sortPosIndex(1:3);
    maxNegIndex = sortNegIndex(1:3);

    PosAlphaIndex =model_svm_linear.('sv_indices')(1:pos);
    maxPosAlphaIndex = PosAlphaIndex(maxPosIndex);
    NegAlphaIndex =model_svm_linear.('sv_indices')(pos+1:end);
    maxNegAlphaIndex = NegAlphaIndex(maxNegIndex);

    maxPosAlphaIndex_c(class_i+1,:) = maxPosAlphaIndex;
    maxNegAlphaIndex_c(class_i+1,:) = maxNegAlphaIndex;

    w = sum(model_svm_linear.('sv_coef').*model_svm_linear.('SVs'));
    b = model_svm_linear.('rho');
    marginTrain(class_i+1,:) = train_labels_i.*(train_imgs*w' + b);
    
    % predict 1
    marginTest(class_i+1,:) = dec_values';
    
    fig1 = figure;
    for i = 1:3
        subplot(1,3,i)
        imshow(reshape(train_imgs(maxPosAlphaIndex_c(class_i+1,i),:),28,28)');
        if i==2
            title(sprintf('digit %d, y=1',class_i));
        end
    end 
    saveas(fig1,sprintf('c%d_dig%d_posSV.png',c,class_i));
      

    fig2 = figure;
    for i = 1:3
        subplot(1,3,i)
        imshow(reshape(train_imgs(maxNegAlphaIndex_c(class_i+1,i),:),28,28)');
        if i==2
            title(sprintf('digit %d, y=-1',class_i));
        end
    end
    saveas(fig2,sprintf('c%d_dig%d_negSV.png',c,class_i));
     

    fig3 = figure;
    cdfplot(marginTrain(class_i+1,:));
    xlabel('margin')
    ylabel('cumulative distribution');
    title(sprintf('digit %d CDF',class_i));
    saveas(fig3,sprintf('c%d_dig%d_margin.png',c,class_i)); 
      
end


%% overall prediction
[~,maxIndex] = max(marginTest);
maxIndex = maxIndex-1;
fprintf('c = %d total error: %f',c,sum(maxIndex'~=test_labels)/10000);
disp('\n');

%% (c)
c = 8;
gamma = 0.0625;
test_accuracy_c = zeros(10,1);
num_sv_c = zeros(10,1);
maxPosAlphaIndex_c = zeros(10,3);
maxNegAlphaIndex_c = zeros(10,3);
marginTrain = zeros(10,20000);
marginTest = zeros(10,10000);
addpath /Users/shiqi/Downloads/271b/hw4/libsvm-3.24/matlab

%%
for class_i = 0:2
    train_labels_i = -ones(size(train_labels));
    train_labels_i(train_labels==class_i) = 1;
    test_labels_i = -ones(size(test_labels));
    test_labels_i(test_labels==class_i) = 1;
    
    config = sprintf('-c %d -g %f',c,gamma);
    model_svm_radial = svmtrain(train_labels_i, train_imgs, config);
    [predict_label, accuracy, dec_values] = svmpredict(test_labels_i, test_imgs, model_svm_radial);
    test_accuracy_c(class_i+1) = accuracy(1);
    num_sv_c(class_i+1) = model_svm_radial.('totalSV');
    nSV = model_svm_radial.('nSV');
    pos = nSV(1);
    [~,maxPosIndex] = maxk(model_svm_radial.('sv_coef')(1:pos),3);
    [~,maxNegIndex] = maxk(model_svm_radial.('sv_coef')(pos+1:end),3);
    
    [~,sortPosIndex] = sort(-model_svm_radial.('sv_coef')(1:pos));
    [~,sortNegIndex] = sort(model_svm_radial.('sv_coef')(pos+1:end));
    maxPosIndex = sortPosIndex(1:3);
    maxNegIndex = sortNegIndex(1:3);

    PosAlphaIndex =model_svm_radial.('sv_indices')(1:pos);
    maxPosAlphaIndex = PosAlphaIndex(maxPosIndex);
    NegAlphaIndex =model_svm_radial.('sv_indices')(pos+1:end);
    maxNegAlphaIndex = NegAlphaIndex(maxNegIndex);

    maxPosAlphaIndex_c(class_i+1,:) = maxPosAlphaIndex;
    maxNegAlphaIndex_c(class_i+1,:) = maxNegAlphaIndex;

    [~, ~, train_dec_values] = svmpredict(train_labels_i, train_imgs, model_svm_radial);
    marginTrain(class_i+1,:) = train_labels_i.*train_dec_values;

    % predict 1

    marginTest(class_i+1,:) = dec_values';
    
    fig1 = figure;
    for i = 1:3
        subplot(1,3,i)
        imshow(reshape(train_imgs(maxPosAlphaIndex_c(class_i+1,i),:),28,28)');
        if i==2
            title(sprintf('digit %d, y=1',class_i));
        end
    end 
    saveas(fig1,sprintf('c%d_dig%d_posSV.png',c,class_i));
      

    fig2 = figure;
    for i = 1:3
        subplot(1,3,i)
        imshow(reshape(train_imgs(maxNegAlphaIndex_c(class_i+1,i),:),28,28)');
        if i==2
            title(sprintf('digit %d, y=-1',class_i));
        end
    end
    saveas(fig2,sprintf('c%d_dig%d_negSV.png',c,class_i));
     

    fig3 = figure;
    cdfplot(marginTrain(class_i+1,:));
    xlabel('margin')
    ylabel('cumulative distribution');
    title(sprintf('digit %d CDF',class_i));
    saveas(fig3,sprintf('c%d_dig%d_margin.png',c,class_i)); 
      
end

%% overall prediction
[~,maxIndex] = max(marginTest);
maxIndex = maxIndex-1;
fprintf('c = %d total error: %f',c,sum(maxIndex'~=test_labels)/10000);
disp('\n');
    
    