import numpy as np
import cv2
import scipy.spatial


def get_color_for_point(point_coords, list_of_colors, list_of_point_centers):
    color = np.array([0.0, 0.0, 0.0])
    distances = scipy.spatial.distance.cdist([point_coords],
                                             list_of_point_centers)[0]
    weights = 1 / (distances + 0.01)
    for ic, c in enumerate(list_of_colors):
        color += (np.array(c) * weights[ic])
    color /= (np.sum(weights))
    return color


def create_2d_color_map(list_of_points, list_of_colors, height, width):
    rgb = np.zeros((height, width, 3)).astype("uint8")
    c_x = int(width / 2)
    c_y = int(height / 2)
    step = 9
    win_size = int((step-1) / 2)
    for i in range(len(list_of_points)):
        rgb[c_y - int(list_of_points[i][1] * height / 2),
            c_x + int(list_of_points[i][0] * width / 2)] = list_of_colors[i]
    for y in range(win_size, height - win_size, step):
        for x in range(win_size, width - win_size, step):
            x_real = (x - width / 2) / (width / 2)
            y_real = (height / 2 - y ) / (height / 2)
            color = get_color_for_point([x_real, y_real], list_of_colors,
                                        list_of_points)
            rgb[y - win_size: y + win_size,
                x - win_size: x + win_size] = color
    bgr = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    return bgr

if __name__ == "__main__":
    colors = {"coral": [255,127,80],
              "pink": [255, 192, 203],
              "orange": [255, 165, 0],
              "blue": [0, 0, 205],
              "green": [0, 205, 0],
              "red": [205, 0, 0],
              "yellow": [204, 204, 0]}
    angry_pos = [-0.8, 0.5]
    fear_pos = [-0.3, 0.8]
    happy_pos = [0.6, 0.6]
    calm_pos = [0.4, -0.5]
    sad_pos = [-0.6, -0.4]
    bgr = create_2d_color_map([angry_pos, fear_pos, happy_pos, calm_pos, sad_pos],
                              [colors["red"],  colors["yellow"], colors["orange"],
                              colors["green"], colors["blue"]], 200, 200)
    cv2.imshow('Signal', bgr)
    ch = cv2.waitKey(10000)
