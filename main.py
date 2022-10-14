import math

import cv2
import sys
import numpy as np


def open_image():
    print("Enter file location to image")
    location = "C:/Users/hogyv/Downloads/chess_table1.jpg"
    img = cv2.imread(cv2.samples.findFile(location))
    if img is None:
        sys.exit("Could not read image")
    return img


def show_picture(img, window_name="default"):
    cv2.imshow(window_name, img)
    cv2.waitKey(0)


def get_lines(img):
    canny = cv2.Canny(img, 200, 400, None, 3)
    lines = cv2.HoughLines(canny, 1, np.pi / 180, 150, None, 0, 0)

    show_picture(canny, "canny")
    output = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
    line_list = []

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
            line_list.append([pt1, pt2])

            cv2.line(output, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

    show_picture(output, "end")
    print(line_list)
    return line_list


# get m from y = mx + c
def get_gradient(lines):
    lines_grad = []
    for line in lines:
        if line[1][0] - line[0][0] == 0:
            m = math.inf
        elif line[1][1] - line[0][1] == 0:
            m = 0
        else:
            m = (line[1][1] - line[0][1]) / (line[1][0] - line[0][0])
        c = line[0][1] - line[0][0] * m
        print(line[0], line[1], m, c)
        lines_grad.append([m, line[0]])
    return lines_grad


def get_intersections(lines):
    pass


if __name__ == '__main__':
    img1 = open_image()
    show_picture(img1)
    lines = get_lines(img1)
    lines = get_gradient(lines)
    get_intersections(lines)
