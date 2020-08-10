function is_valid = check_validity(BW)
[height, width] = size(BW);
y_scan = [round(height * 0.3), round(height * 0.5), round(height * 0.7)];
x_start = round(width * 0.1);
status = [BW(y_scan(1), x_start), BW(y_scan(2), x_start), BW(y_scan(3), x_start)];
n_flip = [0, 0, 0];
for x = x_start:round(width * 0.9)
    for y = 1:length(y_scan)
        if BW(y_scan(y), x) ~= status(y)
            status(y) = BW(y_scan(y), x);
            n_flip(y) = n_flip(y) + 1;
        end
    end
end

is_valid = all(n_flip < 30); 
end
