import cv2 as cv
import numpy as np
import math

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split



digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1)) 
classifier = svm.SVC(gamma=0.001)
X_train, _, y_train,_ = train_test_split(
    data, digits.target, test_size=1, shuffle=False)
classifier.fit(X_train, y_train)
cv.startWindowThread()
cap = cv.VideoCapture('sentry3.mkv')
frames=1
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('output.mp4', fourcc, 10.0, (1440,810),True)

while(frames<29):

    ret, frame = cap.read()
    cv.waitKey(1)
    frames=frames+1
while (frames<179):
    ret, frame = cap.read()
    cv.waitKey(1)
    frames=frames+1
    frame2=frame.copy()
    Mask=cv.inRange(frame2,np.array([0, 0, 0]),np.array([50,50,50]))
    Mask=cv.bitwise_not(Mask)
    kernel = np.ones((5,5),np.uint8)

    dilation = cv.dilate(Mask,kernel,iterations = 1)
    erosion = cv.erode(dilation,kernel,iterations = 13)
    Mask=cv.bitwise_not(erosion)
    contours, hierarchy = cv.findContours(Mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # frame4=np.zeros(frame.shape,dtype=np.int8)
    for i in contours:
        frame4=np.zeros(frame.shape,dtype=np.uint8)
        x1,y1,w1,h1= cv.boundingRect(i)
        
        rect=cv.minAreaRect(i)
  
        box=cv.boxPoints(rect)
   
        box=np.int0(box)
        centre=rect[0]
        cv.rectangle(frame2,(x1,y1),(x1+w1,y1+h1),(20, 255, 57),2)
        cv.drawContours(frame4, [box],0,(255,255,255),-1)
        # cv.drawContours(frame2, [box],0,(20, 255, 57),2)
        roi=cv.bitwise_and(frame,frame4)
        ekkaurmask=cv.inRange(roi,np.array([0, 0,200]),np.array([255,255,255]))
        ekkaurmask = cv.dilate(ekkaurmask,kernel,iterations = 15)
        frame1=cv.bitwise_and(roi,roi,mask=ekkaurmask)
        whitemask=cv.inRange(frame1,np.array([130, 130,130]),np.array([255,255,205]))
        
        contour, _ = cv.findContours(whitemask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if len(contour)!=0:
            c = max(contour, key = cv.contourArea)
            x,y,w,h = cv.boundingRect(c)
            if(w>5 or h>5):
                font = cv.FONT_HERSHEY_COMPLEX 
                imgs=frame[y:y+h, x:x+w]
                framee= cv.resize(imgs,(8,8),interpolation=cv.INTER_LINEAR)
                framee=cv.cvtColor(framee,cv.COLOR_BGR2GRAY)
                framee=framee/16
                test=np.asarray(framee,dtype="int32")
                new=np.asarray(test,dtype="float32")
                predicted = classifier.predict([new.reshape(-1)])
                if ((predicted[0]==1)or (predicted[0]==9) or (predicted[0]==7)):
                    # cv.circle(frame2,, 7, (255,0,0), -1)
                    
                    cv.putText(frame2, 'Red 1', (x1,int (y1-4)), font,0.7, (20, 255, 57),2) 
                if predicted[0]==2:
                    cv.putText(frame2, 'Red 2', (x1,int (y1-4)), font,0.7, (20, 255, 57),2)
                else: 
                    print(box)
                    # cv.drawContours(frame2, [box],0,(255,255,255),2)
        
    cv.imshow('lool',frame2)
    out.write(frame2)
    k = cv.waitKey(1) & 0xff
    if k == 27 : break
 