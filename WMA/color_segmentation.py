import cv2
import numpy as np
import matplotlib.pyplot as plt
import importlib
from matplotlib import colors
from matplotlib.colors import hsv_to_rgb

importlib.import_module("mpl_toolkits.mplot3d").Axes3D

# plik z obiektem do wymaskowania
background_mask = cv2.imread("./data/bg_mask.png")
object = cv2.imread("./data/dach.png")

# plik z obrazem z którego usuwam elementy
background = cv2.imread("./data/background.png")
hsv_background = cv2.cvtColor(background, cv2.COLOR_RGB2HSV)


pixel_colors = object.reshape((np.shape(object)[0] * np.shape(object)[1], 3))
norm = colors.Normalize(vmin=-1.0, vmax=1.0)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()


# obiekt w rzestrzeni hsv
hsv_object = cv2.cvtColor(object, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv_object)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

# wykres do wybrania zakresu
axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
axis.set_zlabel("Value")
plt.show()

# zakresy hsv
lower_color = (110, 50, 75)
upper_color = (145, 150, 175)

# wyswieltenie wybranych kolorów
lo_square = np.full((10, 10, 3), lower_color, dtype=np.uint8) / 255.0
do_square = np.full((10, 10, 3), upper_color, dtype=np.uint8) / 255.0
plt.subplot(1, 2, 1)
plt.imshow(hsv_to_rgb(do_square))
plt.subplot(1, 2, 2)
plt.imshow(hsv_to_rgb(lo_square))
plt.show()


# maskowanie
mask = cv2.inRange(hsv_background, lower_color, upper_color)
result = cv2.bitwise_and(hsv_background, background, mask=mask)
cv2.imshow("original", hsv_background)
cv2.imshow("original", background)
cv2.imshow("mask green color", mask)

cv2.waitKey(0)
