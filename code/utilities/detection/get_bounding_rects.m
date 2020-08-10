function rects = get_bounding_rects(mask, extend_factors)
if nargin < 2
    extend_factors = [1, 1];
end

[height, width] = size(mask);
stats = regionprops(mask, 'BoundingBox');
boxes = {stats.BoundingBox};

rects = {};

for k = 1:length(boxes)  
    rects{k}.point1 = ...
        [max(round(boxes{k}(1) + 0.5 * boxes{k}(3) * (1 - extend_factors(1))), 1), ...
         max(round(boxes{k}(2) + 0.5 * boxes{k}(4) * (1 - extend_factors(2))), 1)];
     
    rects{k}.point2 = ...
        [min(round(boxes{k}(1) + 0.5 * boxes{k}(3) * (1 + extend_factors(1))), width), ...
         min(round(boxes{k}(2) + 0.5 * boxes{k}(4) * (1 + extend_factors(2))), height)];
    
end
end
