function [best_thresh, best_twin, best_index] = decision_stump(x, y, weights)
    thresholds = (0:50)/50;
    [num_sample, dim] = size(x);

    thresh_dim = zeros(dim,1);
    twin_dim = zeros(dim,1);
    weight_pred_y_dim = zeros(dim,1);
    
    for i = 1:dim
        x_ith = x(:,i);
        best_thresh_j = -1;
        best_twin_j = 0;
        max_weight_pred_y = -inf;
        for j = 1:51
            pred = -ones(num_sample,1);
            pred(x_ith>=thresholds(j)) = 1;
           
            if sum(pred==y) < num_sample/2
                twin = -1;
                pred = -pred;
            else
                twin = 1;
                
            end
            weight_pred_y = sum(weights.*pred.*y);
            if weight_pred_y >= max_weight_pred_y
                max_weight_pred_y = weight_pred_y;
                best_thresh_j = thresholds(j);
                best_twin_j = twin;
            end
        end
        thresh_dim(i) = best_thresh_j;
        twin_dim(i) = best_twin_j;
        weight_pred_y_dim(i) = max_weight_pred_y;
        
    end
    [~, best_index] = max(weight_pred_y_dim);
    best_thresh = thresh_dim(best_index);
    best_twin = twin_dim(best_index);
end

