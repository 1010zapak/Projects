#implementation of sticker making article from Machine learning India
import cv2
import numpy as np

img = cv2.imread("C:\\Users\\anubh\\Downloads\\rampage-dwayne-johnson-movie-poster-8k-wallpaper-1024x576.jpg")

num_down = 2
num_bilateral =7
img = cv2.resize(img,(1920//2,1080//2))

img_color = img          #### downsample image using gaussian pyramid
for _ in range(num_down):
    img_color = cv2.pyrDown(img_color)

#### repeatedly apply small bilateral filter instead of one large filter
for _ in range(num_bilateral):
    img_color = cv2.bilateralFilter(img_color,d=9,sigmaColor=9,sigmaSpace=7)

### upsample image, convert 2 grayscale, apply median blur, thresholding
for _ in range(num_down):
    img_color = cv2.pyrUp(img_color)

img_gray = cv2.cvtColor(img_color,cv2.COLOR_RGB2GRAY)
img_blur = cv2.medianBlur(img_gray,7)

img_edge = cv2.adaptiveThreshold(img_blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,blockSize=9,C=2)


###perform bitwise AND and display results
#Reason for this section ???
##############
img_edge = cv2.cvtColor(img_edge,cv2.COLOR_GRAY2RGB)
img_cartoon = cv2.bitwise_and(img_color,img_edge)
###############

#### display
#cv2.imshow("cartoon",img_cartoon)
stack = np.hstack([img,img_cartoon])
cv2.imshow("Stacked images",stack)
cv2.waitKey(0)