function data = smooth_lower_data(data, threshold)
% threshold [low, high]
%   low - 触发 smooth 操作尝试的起始上升点的最大值
%   high - 使 smooth 操作有效的峰值的最大值
rise_start_pos = 0;
is_tracking = 0;
is_searching = 0;  % whether is looking for the second rising edge

for x = 2:length(data)
    derivate = sign(data(x) - data(x - 1));
    if is_searching
        if derivate > 0
            % got the second rising edge, do linear interpolation
            start_val = data(rise_start_pos);
            data(rise_start_pos:x - 1) = linspace(start_val, data(x - 1), x - rise_start_pos);
            is_searching = 0;
            if data(x - 1) < start_val
                rise_start_pos = x - 1;
            end
            if data(x) >= threshold(1)
                is_tracking = 0;
            end
        end
    else
        if ~is_tracking && derivate > 0 && data(x - 1) < threshold(1) && data(x) <= threshold(2)
            % got the first rising edge
            rise_start_pos = x - 1;
            is_tracking = 1;
        elseif is_tracking
            if derivate > 0 && data(x) > threshold(2)
                is_tracking = 0;
            elseif derivate < 0
                is_searching = 1;
            end
        end
    end
end

if is_searching
    % 平滑尾段
    data(rise_start_pos:x) = linspace(data(rise_start_pos), data(x), x - rise_start_pos + 1);
end
end
