function polarity = detect_polarity(BW)
% polarity: 0 - foreground is logical 1; 1 - foreground is logical 0;
BW = medfilt2(BW, [3, 3]);
n_pixel_threshold = round(numel(BW) * 0.08);
BW_background = bwareaopen(BW, n_pixel_threshold);
if sum(BW_background == 0, 'all') > 5 * sum(BW_background == 1, 'all')
    polarity = 0;
    return;
end

BW_background = bwareaopen(~BW, n_pixel_threshold);
if sum(BW_background == 0, 'all') > 5 * sum(BW_background == 1, 'all')
    polarity = 1;
    return;
end

polarity = (sum(BW == 0, 'all') < sum(BW == 1, 'all'));
end
