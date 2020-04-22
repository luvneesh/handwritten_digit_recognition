import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
img=cv.imread('2.png')
frame= cv.resize(img,(8,8),interpolation=cv.INTER_LINEAR)
frame=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
frame=frame/16
test=np.asarray(frame,dtype="int32")
new=np.asarray(test,dtype="float32")
image=new.reshape(-1)
digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1)) 
classifier = svm.SVC(gamma=0.001)
X_train, _, y_train,_ = train_test_split(
    data, digits.target, test_size=1, shuffle=False)
classifier.fit(X_train, y_train)
predicted = classifier.predict([image])
print(predicted)