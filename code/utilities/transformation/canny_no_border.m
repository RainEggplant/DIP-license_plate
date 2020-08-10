function img_out = canny_no_border(img_out, threshold)
img_out = edge(img_out, 'Canny', threshold);
img_out(1:3, :) = 0;
img_out(end - 2:end, :) = 0;
img_out(:, 1:3) = 0;
img_out(:, end - 2:end) = 0;
end
