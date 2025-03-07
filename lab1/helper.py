import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

I = cv2.imread('mandrill.jpg')

fig, ax = plt.subplots(1)
rect = Rectangle((50,50),50,100, fill=False, ec = 'r') # ec = edge color
ax.add_patch(rect)
plt.show()