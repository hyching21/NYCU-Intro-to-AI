import numpy as np
import cv2 

#read image
image = cv2.imread('image.png')

#read .txt file
with open('bounding_box.txt', 'r') as f:
    coordinates = f.readlines()

for coordinate in coordinates:
    x1, y1, x2, y2 = map(int,coordinate.split())
    # draw bounding box
    cv2.rectangle(image,(x1,y1),(x2,y2),(0,0,255),2)

#save image
cv2.imwrite('hw0_111550113_1.png', image)


#show image

#cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
#cv2.imshow("Image", image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

