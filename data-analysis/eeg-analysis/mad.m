function md = mad(v, flag, dim)
% A local implementation of Median Absolute Deviation (MAD) to satisfy
% a toolbox dependency. This version handles up to three input arguments
% (v, flag, dim) for maximum compatibility.

    % If no dimension is specified, find the first non-singleton dimension
    if nargin < 3
        dim = find(size(v) ~= 1, 1);
        if isempty(dim), dim = 1; end
    end

    % Calculate the median along the specified dimension
    med_v = median(v, dim);

    % Calculate the median of the absolute deviations from the median
    md = median(abs(v - med_v), dim);

    % Apply the scaling factor if the flag is provided and set to 1
    if nargin > 1 && ~isempty(flag) && flag == 1
        md = md * 1.4826;
    end

end