%% Using Boosting to make classification
%% load dataset
clc;
clear all;

[train_imgs, train_labels] = readMNIST('train-images.idx3-ubyte','train-labels.idx1-ubyte', 20000, 0);
assert(all(size(train_imgs)==[20000,784]));
assert(all(size(train_labels)==[20000,1]));

[test_imgs, test_labels] = readMNIST('t10k-images.idx3-ubyte','t10k-labels.idx1-ubyte', 10000, 0);
assert(all(size(test_imgs)==[10000,784]));
assert(all(size(test_labels)==[10000,1]));



%% boosting 

[num_sample, dim] = size(train_imgs);
[num_test, ~] = size(test_imgs);

% class_i: 1~10
T = 250;
% 0 1 2 3 4 5 6 7 8 9
class_i = 9;
train_labels_i = -ones(size(train_labels));
train_labels_i(train_labels==class_i) = 1;
test_labels_i = -ones(size(test_labels));
test_labels_i(test_labels==class_i) = 1;

weights = zeros(num_sample,1);
g_train = zeros(num_sample,1);
g_test = zeros(num_test,1);

train_prob_error = zeros(T,1);
test_prob_error = zeros(T,1);

margin = zeros(num_sample,1);
store_margin = zeros(num_sample,5);
store_margin_count = 1;

max_weight_index = zeros(T,1);

for t=1:T
    margin = train_labels_i.*g_train;
    if ismember(t,[5,10,50,100,250])
        store_margin(:,store_margin_count) = margin;
        store_margin_count = store_margin_count +1 ;
    end
        
    weights = exp(-margin);
    
    [best_thresh_t,best_twin_t, best_index_t] = decision_stump(train_imgs, train_labels_i, weights);
    pred_train = alpha_t(train_imgs, best_thresh_t, best_twin_t, best_index_t);
    pred_test = alpha_t(test_imgs, best_thresh_t, best_twin_t, best_index_t);
    epsilon_t = sum(weights(pred_train~=train_labels_i))/sum(weights);
    Wt = 0.5*log((1-epsilon_t)/epsilon_t);
    
    g_train = g_train + Wt.*pred_train;
    g_test = g_test + Wt.*pred_test;
    
    g_pred_train = -ones(num_sample,1);
    g_pred_train(g_train>=0) = 1;
    
    g_pred_test = -ones(num_test,1);
    g_pred_test(g_test>=0) = 1;
    
    train_prob_error(t) = sum(g_pred_train~=train_labels_i)/num_sample;
    test_prob_error(t) = sum(g_pred_test~=test_labels_i)/num_test;
    [~,max_weight_index(t)] = max(weights);
    
end


% plot, save workspace

[num_per_index, which_index] = hist(max_weight_index,unique(max_weight_index));
[~, sort_index] = sort(-num_per_index);
freq_index = which_index(sort_index);
three_most_common_index = freq_index(1:3);
for i = 1:3
    figure;
    imshow(reshape(train_imgs(three_most_common_index(i),:),28,28)');
end

filename = sprintf('digit%d.mat',class_i);
save(filename);
figure;
plot(1:T,train_prob_error,1:T,test_prob_error);

xlabel('number of iterations');
ylabel('probability error');
title(['digit',num2str(class_i)]);
legend('train','test');

figure;

cdfplot(store_margin(:,1));
hold on;
cdfplot(store_margin(:,2));
hold on;
cdfplot(store_margin(:,3));
hold on;
cdfplot(store_margin(:,4));
hold on;
cdfplot(store_margin(:,5));
hold off;
xlabel('margin')
ylabel('cumulative distribution');
legend('5th iter','10th iter','50th iter','100th iter','250ith iter');
title(['digit',num2str(class_i)]);


figure;
plot(1:T,max_weight_index);
xlabel('number of iteration');
ylabel('index of largest weight');
title(['digit',num2str(class_i)]);



