function mask = get_plate_mask(edge_car)
[height, width] = size(edge_car);

img_cl = imclose(edge_car, ones(round(height / 120), round(width / 40)));
img_op = imopen(img_cl, ones(round(height / 120), round(width / 40)));

img_cl = imclose(img_op, ones(round(height / 66), round(width / 22)));
img_op = imopen(img_cl, ones(round(height / 66), round(width / 22)));

img_cl = imclose(img_op, ones(round(height / 42), round(width / 14)));
img_op = imopen(img_cl, ones(round(height / 42), round(width / 14)));

img_op = imopen(img_op, ones(round(height / 24), round(width / 8)));
img_cl = imclose(img_op, ones(round(height / 24), round(width / 8)));

img_op = imopen(img_cl, ones(round(height / 18), round(width / 7)));
img_cl = imclose(img_op, ones(round(height / 18), round(width / 7)));

mask = img_cl;
end
