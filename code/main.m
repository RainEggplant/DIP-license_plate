%%
IMG_FILENAME = '00.jpg';
IMG_PATH = ['images/', IMG_FILENAME];
img = imread(IMG_PATH);

%%
[mask, img_plate_color, img_characters, x_ranges, y_range] = extract_plate_and_characters(img);

%%
n_ch = length(x_ranges);
figure;
sgtitle(IMG_FILENAME);
subplot(3, n_ch, [1:n_ch]);
imshow(img_plate_color);
for n = 1:length(x_ranges)
    subplot(3, n_ch, n_ch + n);
    imshow(img_plate_color(y_range(1):y_range(2), x_ranges{n}(1):x_ranges{n}(2), :));
    subplot(3, n_ch, 2 * n_ch + n);
    imshow(img_characters(y_range(1):y_range(2), x_ranges{n}(1):x_ranges{n}(2)));
end

imwrite(mask, ['output/mask/', IMG_FILENAME]);
