import numpy as np
import cv2
import scipy.spatial

def create_2d_color_map(list_of_points, list_of_colors, height, width):
    rgb = np.zeros((height, width, 3)).astype("uint8")
    c_x = int(width / 2)
    c_y = int(height / 2)
    for i in range(len(list_of_points)):
        rgb[c_y - int(list_of_points[i][1] * height / 2),
            c_x + int(list_of_points[i][0] * width / 2)] = list_of_colors[i]

    for y in range(height):
        for x in range(width):
            color = np.array([0.0, 0.0, 0.0])
            x_real = ((x - width/2)) / (width / 2)
            y_real = ((height/2 -y )) / (height / 2)
            distances = scipy.spatial.distance.cdist([[x_real, y_real]],
                                                     list_of_points)[0]
            weights = 1 / (distances + 1)
            for ic, c in enumerate(list_of_colors):
                color += (np.array(c) * weights[ic])
            color /= (np.sum(weights))
            color = color.astype("uint8")
            rgb[y, x] = color

    bgr = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    cv2.imshow('Signal', bgr)
    ch = cv2.waitKey(10000)

if __name__ == "__main__":
    create_2d_color_map([[-0.5, 0.5], [0.5, 0], [-0.5, -0.5], [-0.2, 0.8], [0.5, 0.5]],
                        [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 127, 80]],
                        200, 200)