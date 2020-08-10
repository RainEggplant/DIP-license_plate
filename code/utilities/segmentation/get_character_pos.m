function [img_characters, x_ranges, y_range] = get_character_pos(img_plate)
img_gray = grayscale(img_plate);
img_eq = adapthisteq(img_gray, 'NumTiles', [4, 6]);
img_bin = imbinarize(img_eq);
polarity = detect_polarity(img_bin);
img_bin = xor(img_bin, polarity);

[img_characters, x_ranges, y_range] = split_character(img_bin);
end
