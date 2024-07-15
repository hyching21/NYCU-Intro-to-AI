import numpy as np
import cv2 

#read image
image = cv2.imread('image.png')
height, width = image.shape[0:2]
h=int(height/2)
w=int(width/2)

#1. translation
M1 = np.float32([[1,0,100],[0,1,180]])
translate_image = cv2.warpAffine(image, M1, (width,height))
translate_image = cv2.resize(translate_image, (w, h))
cv2.imshow("translation", translate_image)

#2. flipping (horizontal & vertical)
flip_image = cv2.flip(image, -1) 
flip_image = cv2.resize(flip_image, (w, h))
cv2.imshow("flipping", flip_image)

#3. cropping
crop_image = image[200:500,200:850]
#crop_image = cv2.resize(crop_image, (w, h))
cv2.imshow("cropping", crop_image)

#4. rotation
M4 = cv2.getRotationMatrix2D((w,h),45,1)
rotate_image = cv2.warpAffine(image, M4, (width,height))
rotate_image = cv2.resize(rotate_image, (w, h))
cv2.imshow("rotation", rotate_image)

#5. Affine transformation
p1 = np.float32([[50,100],[510,0],[0,720]])
p2 = np.float32([[0,0],[510,0],[0,720]])
M3 = cv2.getAffineTransform(p1, p2)
affine_image = cv2.warpAffine(image, M3, (width,height))
affine_image = cv2.resize(affine_image, (w, h))
cv2.imshow("affine transformation", affine_image)


#close window
cv2.waitKey(0)
cv2.destroyAllWindows()




