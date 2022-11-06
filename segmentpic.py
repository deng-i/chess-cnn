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

    # show_picture(canny, "canny")
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

    # show_picture(output, "end")
    return line_list


# get m, c from y = mx + c
def get_gradient(lines):
    lines_grad = []
    for line in lines:
        if line[1][0] - line[0][0] == 0:
            m = 999999999
        elif line[1][1] - line[0][1] == 0:
            m = 0
        else:
            m = (line[1][1] - line[0][1]) / (line[1][0] - line[0][0])
        c = line[0][1] - line[0][0] * m
        lines_grad.append([m, c, line[0]])
    return lines_grad


def get_intersections(lines, img):
    intersections = []
    straightness = 1
    img2 = img.copy()
    while len(lines) > 1:
        current_line = lines.pop()
        for line in lines:
            # two vertical lines should not intersect
            if abs(current_line[0]) > straightness * 2 and abs(line[0]) > straightness * 2:
                continue
            # two horizontal lines should not intersect
            if abs(current_line[0]) < straightness / 2 and abs(line[0]) < straightness / 2:
                continue
            m = current_line[0] - line[0]
            c = current_line[1] - line[1]
            x = -c / m
            y = current_line[0] * x + current_line[1]
            x = round(x)
            y = round(y)
            if abs(y) <= height and abs(x) <= width:  # intersection is in the picture
                intersections.append((x, y))
                cv2.circle(img2, (x, y), 0, (0, 0, 255), 5)
    # show_picture(img2, "points")
    intersections.sort(reverse=True)  # should speed up clearing
    return intersections


def clear_intersections(intersections, img):
    new_intersections = []
    img2 = img.copy()

    def dist(i1, i2):
        distance = ((i1[0] - i2[0]) ** 2 + (i1[1] - i2[1]) ** 2) ** 0.5
        return distance

    # remove close intersections
    while len(intersections) > 1:
        flag = True
        intersection = intersections.pop()
        for inters in intersections:
            # there is another close intersection, so this one is not needed
            if dist(intersection, inters) < (height / 16):
                flag = False
                break
        if flag:
            new_intersections.append(intersection)
            cv2.circle(img2, intersection, 0, (0, 0, 255), 5)
    # last intersection has to be added manually
    new_intersections.append(intersections[0])
    cv2.circle(img2, intersections[0], 0, (0, 0, 255), 5)

    new_intersections.sort()
    show_picture(img2, "cleared")
    return new_intersections


def chop_image(intersections, img):
    # img2 = img.copy()
    x_avg = [[] for _ in range(9)]
    y_avg = [[] for _ in range(9)]
    # there will be 9x9 points
    j = 0
    for x, y in intersections:
        x_avg[j // 9].append(x)
        y_avg[j % 9].append(y)
        j += 1
    for i in range(9):
        x_avg[i] = round(np.mean(x_avg[i]))
        y_avg[i] = round(np.mean(y_avg[i]))

    # img2 = img2[x_avg[0]:x_avg[8], y_avg[0]:y_avg[8]]
    # show_picture(img2, "cropped")
    # will start at top left, go left to right, top to bottom
    tiles = [[[] for _ in range(8)] for _ in range(8)]
    for i in range(8):
        for j in range(8):
            print(x_avg[i], x_avg[i+1], y_avg[j], y_avg[j+1])
            tiles[i][j] = img[x_avg[i]:x_avg[i + 1], y_avg[j]:y_avg[j + 1]]
    return tiles


if __name__ == '__main__':
    img1 = open_image()
    height = img1.shape[0]
    width = img1.shape[1]
    show_picture(img1)
    lines = get_lines(img1)
    lines = get_gradient(lines)
    intersections = get_intersections(lines, img1)
    intersections = clear_intersections(intersections, img1)
    tiles = chop_image(intersections, img1)
