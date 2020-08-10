function H_valid = get_H_valid(H, theta_index, offset)
n_index = size(H, 2);
H_valid = H;
index_r = theta_index + offset;
index_l = theta_index - offset;
if index_r >= n_index
    H_valid(:, mod(index_r, n_index) + 1:index_l - 1) = 0;
elseif index_l <= 1
    H_valid(:, index_r + 1:mod(index_l - 2, n_index) + 1) = 0;
else
    H_valid(:, 1:index_l - 1) = 0;
    H_valid(:, index_r + 1:end) = 0;
end
end
