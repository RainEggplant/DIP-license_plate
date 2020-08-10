function mask = in_range(img, lower, upper)
% lower: Define three element vector here for each colour plane i.e. [0 128 128];
% upper: Define three element vector here for each colour plane i.e. [0 128 128];

mask = true(size(img, 1), size(img, 2));
for p = 1 : 3
    mask = mask & (img(:, :, p) >= lower(p) & img(:, :, p) <= upper(p));
end
end
