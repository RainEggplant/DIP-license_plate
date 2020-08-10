function index = get_hline_theta_index(H)
H_max = max(H, [], 'all');
H_valid = H .* (H > 0.25 * H_max);
theta_hist = sum(H_valid, 1);
[val1, index1] = max(theta_hist(1:10));
[val2, index2] = max(theta_hist(end - 9:end));
if val1 > val2
    index = index1;
else
    index = length(theta_hist) - (10 - index2);
end
end
