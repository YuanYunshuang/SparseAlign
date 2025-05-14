

import numpy as np


# Returns the minimum (closest) depth for a specified radius around the center
def depth_min(depths, center, r=10) -> float:
    selected_depths = depths[circular_mask(len(depths), center, r)]
    filtered_depths = selected_depths[(0 < selected_depths) & (selected_depths < 1)]
    if 0 in depths:  # Check if cursor is at widget border
        return 1
    if len(filtered_depths) > 0:
        return np.min(filtered_depths)
    else:
        return 1


# Creates a circular mask with radius around center
def circular_mask(arr_length, center, radius):
    dx = np.arange(arr_length)
    dx2 = (dx[np.newaxis, :] - center) ** 2 + \
          (dx[:, np.newaxis] - center) ** 2
    return dx2 < radius ** 2