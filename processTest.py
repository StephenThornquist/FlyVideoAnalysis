import numpy as np
import cv2

cap = cv2.VideoCapture('/Users/stephen/Desktop/FlyVideoAnalysis/Male.mp4')
cv2.namedWindow("input")
# Video codec
#fourcc = cv2.cv.CV_FOURCC('M','J','P','G')
ret, frameMinusOne = cap.read()
#out = cv2.VideoWriter('curves.avi',fourcc, 30.0, (1920,1080))
# start a background subtractor
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
fgbg = cv2.BackgroundSubtractorMOG()
alpha = .8
maskMinusOne = fgbg.apply(frameMinusOne)
contoursMO, hierMO = cv2.findContours(maskMinusOne, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Parameters for a fly
minFlyArea = 8
maxFlyArea = 150
approxEllip = 8*np.pi
i =1
while(1):
    # Read new frame
    ret, frame = cap.read()
    frame = cv2.resize(frame,(800,450))
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    fgkern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN,fgkern)
    blahbg = cv2.dilate(gray, kernel, iterations = 2)
    dist = cv2.distanceTransform(gray,cv2.cv.CV_DIST_L2,0)
    #ret, blahfg = cv2.threshold(dist,.7*dist.max(),255,0)
    cv2.imshow('frame',gray)
    grayThresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,13,0)
    bkgCnt, hier = cv2.findContours(grayThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # create foreground mask
    fgmask = fgbg.apply(frame)
    # add the contours in for the background outlines
    for cnt in bkgCnt:
        per = cv2.arcLength(cnt,1)
        area = cv2.contourArea(cnt)
        if per > 200 and  (per*per/area < 12*4*np.pi):
            cv2.drawContours(frame, [cnt], 0, (0,255,0), 2)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN,kernel)
    # Find contours in the foreground
    contours, hier = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        per = cv2.arcLength(cnt,1)
        # is it a fly?
        if (minFlyArea < area < maxFlyArea) and (per*per/area < approxEllip) :
            cv2.drawContours(frame, [cnt],0, (255,0,0),2)
            cv2.drawContours(fgmask, [cnt], 0, -255, -1)
        # Get rid of small speckles
        elif area < minFlyArea:
            cv2.drawContours(fgmask, [cnt], 0, 255, -1)
        # Get rid of noncircular contours that are not flies
        elif area > maxFlyArea and (.5*4*np.pi > per*per/area or per*per/area > 2*4*np.pi):
            cv2.drawContours(fgmask, [cnt], 0, 255, -1)
    # Add a red mask for the contours from the lights
    redmask = np.zeros(fgmask.shape + (3,), np.uint8)
    redmask[:,:,2] = 255*fgmask
    redmask[:,:,1]=np.zeros(fgmask.shape)
    redmask[:,:,0]=np.zeros(fgmask.shape)
    # Combine the grayscale image with the 
    draw = cv2.addWeighted(frame,alpha, redmask, 1-alpha,0)
#    out.write(resized)
    #cv2.imshow('frame',draw)
    frameMinusOne = frame
    maskMinusOne = fgmask
    contoursMO = contours
    hierMO = hier
    k=cv2.waitKey(3) 
    print "read frame " + str(i)
    i = i+1
    if k == 27:
        break

cap.release()
#out.release()
cv2.destroyAllWindows()
