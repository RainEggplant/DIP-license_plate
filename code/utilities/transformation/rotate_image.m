function img_out = rotate_image(img_in, rot_angle, method)
if nargin < 3
    method = 'bicubic';
end

height = size(img_in, 1);
width = size(img_in, 2);
img_rot = imrotate(img_in, rot_angle, method, 'crop');

alpha = atand(width / height);
half_diag = sqrt(width^2 + height^2) / 2;

rot_angle = abs(rot_angle);
w1 = half_diag * sind(alpha + rot_angle) - width / 2;
h1 = height / 2 - half_diag * cosd(alpha + rot_angle);
h_cut = ceil(h1 - w1 * tand(rot_angle));

h2 = half_diag * cosd(alpha - rot_angle) - height / 2;
w2 = width / 2 - half_diag * sind(alpha - rot_angle);
w_cut = ceil(w2 - h2 * tand(rot_angle));

img_out = img_rot(h_cut + 1:height - h_cut, w_cut + 1:width - w_cut, :);
end
