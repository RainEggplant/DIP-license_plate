function projection = calc_projection(BW)

[height, width] = size(BW);

%% version weighted
weight = normpdf([1:height], (1 + height) / 2, height / 6);
sum_along_x = zeros(1, width);
for x = 1:width
    sum_along_x(x) = weight * BW(:, x);
end

projection = smooth_lower_data(sum_along_x, [0.06, 0.08]);

%% version cut
% cut_size = round(height * 0.15);
% img_cut = BW(cut_size:height - cut_size, :);
% sum_along_x = sum(img_cut, 1);
% th = 0.08 * max(sum_along_x);
% sum_along_x_smooth = smooth_lower_data(sum_along_x, [th , 1.5 * th]);

end
