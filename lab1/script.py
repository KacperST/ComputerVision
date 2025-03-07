# %%
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib 
import numpy as np

# %%
I = cv2.imread('mandril.jpg')
cv2.imshow("Mandril",I)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
cv2.imwrite("m.png",I) 

# %%
I

# %%
print(I.shape) # dimensions /rows, columns, depth/
print(I.size) # number of bytes
print(I.dtype) # data type

# %%
I = plt.imread('mandril.jpg')

# %%
plt.figure(1) # create figure
plt.imshow(I) # add image
plt.title('Mandril') # add title
plt.axis('off') # disable display of the coordinate system
plt.show() # display

# %%
plt.imsave('mandril.png',I)

# %%
x = [ 100, 150, 200, 250]
y = [ 50, 100, 150, 200]
plt.plot(x,y,'g.',markersize=20)

# %%
fig, ax = plt.subplots(1)
plt.xlim(0, 120)
plt.ylim(0, 120)
rect = Rectangle((10,10),40,40, fill=True, ec = 'r') # ec = edge color
ax.add_patch(rect)
plt.show()

# %%
IG = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
IHSV = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)

plt.figure(1)
plt.imshow(IG, cmap='gray')
plt.title('Mandril in gray scale')
plt.axis('off')
plt.show()


# %%
plt.figure(1)
plt.imshow(IHSV)
plt.title('Mandril in HSV')
plt.axis('off')
plt.show()


# %%
IH = IHSV[:,:,0]
plt.figure(1)
plt.imshow(IH, cmap='hsv')
plt.title('Mandril in H')
plt.axis('off')
plt.show()

# %%
IS = IHSV[:,:,1]
plt.figure(1)
plt.imshow(IS, cmap='hsv')
plt.title('Mandril in S')
plt.axis('off')
plt.show()

# %%
IV = IHSV[:,:,2]
plt.figure(1)
plt.imshow(IV, cmap='hsv')
plt.title('Mandril in V')
plt.axis('off')
plt.show()


# %%
def rgb2gray(I):
    return 0.299*I[:,:,0] + 0.587*I[:,:,1] + 0.114*I[:,:,2]

# %%
IG = rgb2gray(I)
plt.figure(1)
plt.imshow(IG, cmap='gray')
plt.title('Mandril in gray scale')
plt.axis('off')
plt.show()

# %%
_HSV = matplotlib.colors.rgb_to_hsv(I)
plt.figure(1)
plt.imshow(_HSV, cmap='hsv')
plt.title('Mandril in HSV')
plt.axis('off')
plt.show()


# %%
height, width = I.shape[:2]
scale = 1.75
Ix2 = cv2.resize(I, (int(width*scale), int(height*scale)))
fig, ax = plt.subplots(1,2, figsize=(10,5))
ax[0].imshow(I)
ax[0].set_title('Mandril original')
ax[1].imshow(Ix2)
ax[1].set_title('Mandril scaled')
plt.show()


# %%
lena = cv2.imread('lena.png')
lena_gray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)
mandril = cv2.imread('mandril.jpg')
mandril_gray = cv2.cvtColor(mandril, cv2.COLOR_BGR2GRAY)
fig, ax = plt.subplots(2,2  , figsize=(10,10))
ax[0,0].imshow(lena_gray, cmap='gray')
ax[0,0].set_title('Lena')
ax[0,0].axis('off')
ax[0,1].imshow(mandril_gray, cmap='gray')
ax[0,1].set_title('Mandril')
ax[0,1].axis('off')
ax[1, 0].imshow(lena)
ax[1, 0].set_title('Lena')
ax[1, 0].axis('off')
ax[1, 1].imshow(mandril)
ax[1, 1].set_title('Mandril')
ax[1, 1].axis('off')
plt.show()

# %%
lena_plus_madril = lena_gray + mandril_gray
plt.figure(1)
plt.imshow(lena_plus_madril, cmap='gray')
plt.axis('off')
plt.show()


# %%
lena_minus_madril = np.abs(lena_gray - mandril_gray)
plt.figure(1)
plt.imshow(lena_minus_madril, cmap='gray')
plt.axis('off')
plt.show()


# %%
lena_times_madril = lena_gray.astype('float32') * mandril_gray.astype('float32')
plt.figure(1)
plt.imshow(lena_times_madril, cmap='gray')
plt.axis('off')
plt.show()

# %%
def hist(I):
    h = np.zeros((256,1), np.float32)
    height, width = I.shape[:2]
    for y in range(height):
        for x in range(width):
            h[I[y,x]] += 1
    return h

# %%
a = hist(mandril_gray)
plt.figure(1)
plt.plot(a)
plt.show()


# %%
b = cv2.calcHist([mandril_gray], [0], None, [256], [0,256])
plt.figure(1)
plt.plot(b)
plt.show()

# %%
IGE = cv2.equalizeHist(mandril_gray)
IGE_hist = cv2.calcHist([IGE], [0], None, [256], [0,256])
plt.figure(1)
plt.plot(IGE_hist)
plt.show()

# %%
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
I_CLAHE = clahe.apply(mandril_gray)
I_CLAHE_hist = cv2.calcHist([I_CLAHE], [0], None, [256], [0,256])
plt.figure(1)
plt.plot(I_CLAHE_hist)
plt.show()


# %%
fig, ax = plt.subplots(1,5, figsize=(15,5))
mandril_gray_gaussian = cv2.GaussianBlur(mandril_gray, (5,5), 0)
ax[0].imshow(mandril_gray, cmap='gray')
ax[0].set_title('Mandril')
ax[0].axis('off')
ax[1].imshow(mandril_gray_gaussian, cmap='gray')
ax[1].set_title('Mandril Gaussian')
ax[1].axis('off')
mandril_gray_median = cv2.medianBlur(mandril_gray, 5)
ax[2].imshow(mandril_gray_median, cmap='gray')
ax[2].set_title('Mandril Median')
ax[2].axis('off')
mandril_gray_sobel = cv2.Sobel(mandril_gray, cv2.CV_64F, 1, 1, ksize=3)
ax[3].imshow(mandril_gray_sobel, cmap='gray')
ax[3].set_title('Mandril Sobel')
ax[3].axis('off')
mandril_gray_lapsjan = cv2.Laplacian(mandril_gray, cv2.CV_64F)
ax[4].imshow(mandril_gray_lapsjan, cmap='gray')
ax[4].set_title('Mandril Lapsjan')
ax[4].axis('off')
plt.show()



# %%



