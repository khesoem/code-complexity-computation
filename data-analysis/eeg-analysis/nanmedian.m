function y = nanmedian(x)
% A local implementation of nanmedian to satisfy a toolbox dependency.
% This function calculates the median of a vector, ignoring NaN values.

    % Remove any NaN values from the input vector
    x = x(~isnan(x));

    % Calculate the median of the remaining values
    y = median(x);

end