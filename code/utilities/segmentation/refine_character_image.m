function img_refined = refine_character_image(img, polarity)
img_gray = grayscale(img);
img_eq = adapthisteq(img_gray);
img_bin = imbinarize(img_eq);
img_bin = xor(img_bin, polarity);
img_refined = medfilt2(img_bin, [3, 3]);
img_refined = imopen(img_refined, ones(2, 2));
end
