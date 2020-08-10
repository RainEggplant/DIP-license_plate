function lines = get_vertical_lines(img_edge, H_vertical, T, R)
scale = size(H_vertical, 1);
P_vertical = houghpeaks(H_vertical, 30, 'threshold', ceil(0.30 * max(H_vertical(:))));
lines = houghlines(img_edge, T, R, P_vertical, 'FillGap', ...
    scale / 250, 'MinLength', scale / 85);
% scale / 330 for real vertical lines
end
