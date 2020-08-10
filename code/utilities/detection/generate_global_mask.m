function global_mask = generate_global_mask(size_img, mask)
% This is a dirty yet simple and effective method.
mask_value = reshape([1:size_img(1) * size_img(2)], size_img);

[p_tl_y, p_tl_x] = find(mask_value == mask(1, 1));
[p_tr_y, p_tr_x] = find(mask_value == mask(1, end));
[p_bl_y, p_bl_x] = find(mask_value == mask(end, 1));
[p_br_y, p_br_x] = find(mask_value == mask(end, end));

x = [p_tl_x(1), p_tr_x(1), p_br_x(1), p_bl_x(1), p_tl_x(1)];
y = [p_tl_y(1), p_tr_y(1), p_br_y(1), p_bl_y(1), p_tl_y(1)];

global_mask = poly2mask(x, y, size_img(1), size_img(2));
end
