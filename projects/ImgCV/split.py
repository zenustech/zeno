import cv2
import numpy as np

def rgb2hsv(r, g, b):
    """
    Arguments:
    r -- the red value of the RGB color (0-255)
    g -- the green value of the RGB color (0-255)
    b -- the blue value of the RGB color (0-255)

    Returns:
    h -- the hue value of the HSV color (0-360)
    s -- the saturation value of the HSV color (0-1)
    v -- the value (brightness) value of the HSV color (0-1)
    """

    # convert RGB values to percentage
    r, g, b = r / 255.0, g / 255.0, b / 255.0

    # find the minimum and maximum values of the RGB components
    cmin, cmax = min(r, g, b), max(r, g, b)
    delta = cmax - cmin

    # calculate the hue component
    if delta == 0:
        h = 0
    elif cmax == r:
        h = ((g - b) / delta) % 6
    elif cmax == g:
        h = (b - r) / delta + 2
    else:
        h = (r - g) / delta + 4

    h = round(h * 60)

    # calculate the saturation component
    if cmax == 0:
        s = 0
    else:
        s = delta / cmax

    # calculate the value (brightness) component
    v = cmax

    # return the HSV components
    return h, s, v

# Load image
img = cv2.imread('/home/bate/Downloads/googleearthpic.png')

# Convert to HSV
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define color range (blue)
medium = np.array([102, 50, 33])
lower = medium - 30
upper = medium + 30
mask = cv2.inRange(img, lower, upper)
smooth_kernel = np.array([[0,1,0],[1,4,1],[0,1,0]])
for i in range(10):
    for j in range(6):
        mask = cv2.filter2D(mask, -1, smooth_kernel / np.sum(smooth_kernel))
    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
morphs = [1, 2, 3, 4]
for i in morphs:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * i + 1, 2 * i + 1))
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 3), cv2.MORPH_OPEN, kernel, 3)
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 3), cv2.MORPH_CLOSE, kernel, 3)
for i in reversed(morphs):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * i + 1, 2 * i + 1))
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 3), cv2.MORPH_OPEN, kernel, 3)
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 3), cv2.MORPH_CLOSE, kernel, 3)
cv2.imshow('img_out', cv2.bitwise_and(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), img))
cv2.waitKey(0)

contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    color = (0, 255, 0)
    cv2.drawContours(img, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
cv2.imshow('img_out', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # Apply connected component labeling
# num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
#
# # Filter out small components
# min_size = 100
# img_out = img.copy()
# for i in range(1, num_labels):
#     if stats[i, cv2.CC_STAT_AREA] >= min_size:
#         x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
#         cv2.rectangle(img_out, (x, y), (x+w, y+h), (0, 0, 255), 2)
#
# # Display result
# cv2.imshow('img_out', img_out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
