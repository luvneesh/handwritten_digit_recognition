import cv2 as cv
import numpy as np
import math

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
# img=cv.imread('2.png')
'''
pre processing 
frame= cv.resize(img,(8,8),interpolation=cv.INTER_LINEAR)
frame=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

frame=frame/16
test=np.asarray(frame,dtype="int32")
new=np.asarray(test,dtype="float32")
image=new.reshape(-1)
'''
digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1)) 
classifier = svm.SVC(gamma=0.001)
X_train, _, y_train,_ = train_test_split(
    data, digits.target, test_size=1, shuffle=False)
classifier.fit(X_train, y_train)
# predicted = classifier.predict([image])
# print(predicted)

cv.startWindowThread()
cap = cv.VideoCapture('sentry3.mkv')
frames=1
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('output.mp4', fourcc, 20.0, (1440,810),True)
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
    # cv.imshow('b',Mask)
    contours, hierarchy = cv.findContours(Mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    frame4=np.zeros(frame.shape,dtype=np.uint8)
    for i in contours:
        x,y,w,h= cv.boundingRect(i)
       
        rect=cv.minAreaRect(i)
  
        box=cv.boxPoints(rect)
   
        box=np.int0(box)
        centre=rect[0]
        cv.drawContours(frame4, [box],0,(255,255,255),-1)
        cv.drawContours(frame2, [box],0,(255,255,255),2)
        # cv.circle(frame3,(int(centre[0]), 200, 2, (77,93,100), 2)
        # print(ce
    # cv.imshow('lls',frame2)
    frame=cv.bitwise_and(frame,frame4)
    ekkaurmask=cv.inRange(frame,np.array([0, 0,200]),np.array([255,255,255]))
    # ekkaurmask = cv.erode(ekkaurmask,kernel,iterations = 1)
    ekkaurmask = cv.dilate(ekkaurmask,kernel,iterations = 15)
    frame1=cv.bitwise_and(frame,frame,mask=ekkaurmask)
    whitemask=cv.inRange(frame1,np.array([130, 130,130]),np.array([255,255,205]))
    # whitemask=cv.erode(white,kernel,iterations = 1)
    # frame1=cv.bitwise_and(frame1,frame1,mask=whitemask)
    contours, hierarchy = cv.findContours(whitemask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for i in contours:
        x,y,w,h = cv.boundingRect(i)
        if(w>5 or h>5):
            cv.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
            img=frame[y:y+h, x:x+w]
            # cv.imshow('tbc'+str(i),img)
            framee= cv.resize(img,(8,8),interpolation=cv.INTER_LINEAR)
            framee=cv.cvtColor(framee,cv.COLOR_BGR2GRAY)

            framee=framee/16
            test=np.asarray(framee,dtype="int32")
            new=np.asarray(test,dtype="float32")
            new.reshape(-1)
        # rect=cv.minAreaRect(i)
        # print (rect)
        # box=cv.boxPoints(rect)
        # box=np.int0(box)
        # print(box)
        # cv.drawContours(frame1, [box],0,(0,0,255),2)
    cv.imshow('lool',frame1)
    # print(frame1.shape)   
    # out.write(frame1)

    # cv.waitKey(0)
    k = cv.waitKey(1) & 0xff
    if k == 27 : break
    # # yo = cv.erode(gray, None, iterations=1)
    # yo = cv.dilate(yo, None, iterations=1)
    # cnts = np.array(cnts).reshape((-1,1,2)).astype(np.int32)
    # cnts = cv.findContours(yo.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    # #print(cnts)i
    # # for i in range(15):
    #     cv.drawContours(yo,cnts[i],-1,(0,255,0),3)