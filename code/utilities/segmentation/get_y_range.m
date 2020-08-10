function range = get_y_range(BW)
[height, width] = size(BW);
upper_part = BW(1:round(height * 0.15), :);
lower_part = BW(round(height * 0.85):end, :);

upper_proj = sum(upper_part, 2);
lower_proj = sum(lower_part, 2);

y_upper_candidate = find(upper_proj < max(upper_proj) * 0.16);
if isempty(y_upper_candidate)
    y_upper = 1;
else
    y_upper = max(y_upper_candidate);
end

y_lower_candidate = find(lower_proj < max(lower_proj) * 0.16);
if isempty(y_lower_candidate)
    y_lower = height;
else
    y_lower = min(y_lower_candidate) + round(height * 0.85) - 1;
end

range = [y_upper, y_lower];
end
