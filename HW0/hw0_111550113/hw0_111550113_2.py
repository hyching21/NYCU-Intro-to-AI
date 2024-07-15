import numpy as np
import cv2 

#read video and size
video = cv2.VideoCapture("video.mp4")
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))


last_frame = None
while True:
    retval, frame = video.read()
    if not retval:
        break

    #turn gray to get mask
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if(last_frame is None): 
        last_frame = frame_gray   
    diff = cv2.absdiff(frame_gray, last_frame)
    last_frame = frame_gray

    (t,mask) = cv2.threshold(diff,40,255,cv2.THRESH_BINARY)

    foreground = cv2.bitwise_and(frame, frame, mask=mask)
    foreground[:, :, 0] =  0  # Set blue channel to 0
    foreground[:, :, 2] =  0  # Set red channel to 0
    # foreground[:, :, 0] =  np.zeros([diff.shape[0], diff.shape[1]])  # Set blue channel to 0
    # foreground[:, :, 2] =  np.zeros([diff.shape[0], diff.shape[1]])  # Set red channel to 0
    
    result = cv2.hconcat([frame, foreground])
    #result = cv2.resize(result, (width, height))
    cv2.namedWindow('Background Subtraction', cv2.WINDOW_NORMAL)
    cv2.imshow('Background Subtraction', result)

    #esc to close window
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

video.release()
cv2.destroyAllWindows()