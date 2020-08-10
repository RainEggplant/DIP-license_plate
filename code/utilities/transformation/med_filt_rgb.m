function img_out = med_filt_rgb(img_in)
r = medfilt2(img_in(:, :, 1), [3, 3]);
g = medfilt2(img_in(:, :, 2), [3, 3]);
b = medfilt2(img_in(:, :, 3), [3, 3]);
img_out = cat(3, r, g, b);
end