import cv2
import matplotlib.pyplot as plt
import numpy as np


def detect_edges(image, low_threshold=50, high_threshold=150):
    return cv2.Canny(image, low_threshold, high_threshold)


def hough_lines(image):
    """
    `image` should be the output of a Canny transform.

    Returns hough lines (not the image with lines)
    """
    return cv2.HoughLinesP(image, rho=1, theta=np.pi / 180, threshold=5, minLineLength=20, maxLineGap=300)


def average_slope_intercept(lines):
    left_lines = []  # (slope, intercept)
    left_weights = []  # (length,)
    right_lines = []  # (slope, intercept)
    right_weights = []  # (length,)
    # print(lines)
    for line in lines:
        try:
            for x1, y1, x2, y2 in line.squeeze().tolist():
                if x2 == x1:
                    continue  # ignore a vertical line
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
                if slope < 0:  # y is reversed in image
                    left_lines.append((slope, intercept))
                    left_weights.append((length))
                else:
                    right_lines.append((slope, intercept))
                    right_weights.append((length))
        except:
            pass

    solid_or_dotted = [1, 1]  # 第一个为左线，第二个为右线, 默认为实线1
    if len(left_lines) > 1:
        solid_or_dotted[0] = 0
    elif len(right_lines) > 1:
        solid_or_dotted[0] = 0
    # add more weight to longer lines
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None

    return left_lane, right_lane, solid_or_dotted  # (slope, intercept), (slope, intercept)


def make_line_points(y1, y2, line):
    """
    Convert a line represented in slope and intercept into pixel points
    """
    if line is None:
        return None

    slope, intercept = line

    # make sure everything is integer as cv2.line requires it
    try:
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
    except:
        if slope > 0 :
            x2 = 1080 // 2
            x1 = 1080
        else:
            # 斜率小于零，是左边的线
            x1 = 0
            x2 = 1080 // 2

    y1 = int(y1)
    y2 = int(y2)

    return [[x1, y1], [x2, y2]]


def lane_lines(image, lines):
    left_lane, right_lane, solid_or_dotted = average_slope_intercept(lines)

    y1 = image.shape[0]  # bottom of the image
    y2 = y1 * 0.6  # slightly lower than the middle

    left_line = make_line_points(y1, y2, left_lane)
    right_line = make_line_points(y1, y2, right_lane)

    return [left_line, right_line], solid_or_dotted


def draw_lane_lines(image, lines, solid_or_dotted, color=[255, 0, 0], thickness=20):
    # make a separate image to draw lines and combine with the orignal later
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line, color, thickness)
    font = cv2.FONT_HERSHEY_SIMPLEX
    dic = {0: "dotted", 1: "solid"}
    if lines[0] is not None:
        cv2.putText(line_image, dic[solid_or_dotted[0]] + " left lane", (10, 80), font, 3, (255, 0, 0), 5)
    if lines[1] is not None:
        cv2.putText(line_image, dic[solid_or_dotted[1]] + " right lane", (10, 220), font, 3, (255, 0, 0), 5)
    # image1 * α + image2 * β + λ
    # image1 and image2 must be the same shape.
    # cv2.resize(line_image, (1920, 1080))  # 歪了
    lined_img = cv2.addWeighted(image, 1.0, line_image, 0.95, 0.0)
    # plt.ion()
    # plt.figure()
    # plt.imshow(lined_img)
    # plt.show()
    # plt.pause(1)
    # plt.close()
    return lined_img


def lane_fitting(image, seg_imgs):
    list_of_lines = list(map(hough_lines, seg_imgs))
    lines, solid_or_dotted = lane_lines(image, list_of_lines)
    res_img = draw_lane_lines(image, lines, solid_or_dotted)
    return res_img, lines, solid_or_dotted


def get_lane_slope(seg_imgs):
    list_of_lines = list(map(hough_lines, seg_imgs))
    left_lane, right_lane, _ = average_slope_intercept(list_of_lines)
    left_or_right = [0, 0]
    if left_lane is not None:
        left_or_right[0] = 1
    elif right_lane is not None:
        left_or_right[1] = 1
    return left_or_right

# # 利用opencv中的fitLine
# def Cal_kb_linear_fitline(data_line1):
#     loc = [] # 坐标
#     for line in data_line1:
#         x1, y1, x2, y2 = line[0]
#         loc.append([x1,y1])
#         loc.append([x2,y2])
#     loc = np.array(loc) # loc 必须为矩阵形式，且表示[x,y]坐标
#
#     output = cv2.fitLine(loc, cv2.DIST_L2, 0, 0.01, 0.01)
#
#     k = output[1] / output[0]
#     b = output[3] - k * output[2]
#
#     return k,b
