from initialize import *

def make_spiral(x, y):
    # making a spiral
    x_min = 0
    x_max = len(np.unique(x)) - 1
    y_min = 0
    y_max = len(np.unique(y)) - 1
    coord = np.array([], dtype='int')
    y_right = np.arange(y_min, y_max + 1, 1)
    x_right = x_min * np.ones((len(y_right), 1))
    coord_right = np.hstack((x_right, np.array([y_right]).T))

    x_down = np.arange(x_min, x_max, 1) + 1
    y_down = y_max * np.ones((len(x_down), 1))
    coord_down = np.hstack((np.array([x_down]).T, y_down))

    y_left = np.arange(y_max, y_min, -1) - 1
    x_left = x_max * np.ones((len(y_left), 1))
    coord_left = np.hstack((x_left, np.array([y_left]).T))
    x_min = x_min + 1

    x_up = np.arange(x_max, x_min, -1) - 1
    y_up = y_min * np.ones((len(x_up), 1))
    coord_up = np.hstack((np.array([x_up]).T, y_up))
    y_max = y_max - 1

    coord = np.vstack((coord_right, coord_down, coord_left, coord_up))

    while x_min <= x_max and y_min <= y_max:
        y_right = np.arange(y_min, y_max, 1) + 1
        x_right = x_min * np.ones((len(y_right), 1))
        coord_right = np.hstack((x_right, np.array([y_right]).T))
        x_max = x_max - 1

        x_down = np.arange(x_min, x_max, 1) + 1
        y_down = y_max * np.ones((len(x_down), 1))
        coord_down = np.hstack((np.array([x_down]).T, y_down))
        y_min = y_min + 1

        y_left = np.arange(y_max, y_min, -1) - 1
        x_left = x_max * np.ones((len(y_left), 1))
        coord_left = np.hstack((x_left, np.array([y_left]).T))
        x_min = x_min + 1

        x_up = np.arange(x_max, x_min, -1) - 1
        y_up = y_min * np.ones((len(x_up), 1))
        coord_up = np.hstack((np.array([x_up]).T, y_up))
        y_max = y_max - 1

        coord = np.vstack((coord, coord_right, coord_down, coord_left, coord_up))

    coord = np.array(coord, dtype='int')
    return coord[::-1, :]

