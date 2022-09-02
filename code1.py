import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time

#Setting an HTTPS Context to fetch data from OpenML
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context

X = np.load('image.npz')['arr_0']
y = pd.read_csv("https://raw.githubusercontent.com/whitehatjr/datasets/master/C%20122-123/labels.csv")["labels"]
print(y.value_counts())
classes=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
nclasses = len(classes)
print(nclasses)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 2500, train_size = 7500, random_state = 0)
x_train_scale = x_train/255 
x_test_sclae = x_test/255
clf = LogisticRegression(solver = "saga", multi_class = "multinomial")
clf.fit(x_train_scale,y_train)
y_predict = clf.predict(x_test_sclae)
a = accuracy_score(y_test,y_predict)
print("The accuracy is: ", a)

cap = cv2.VideoCapture(0)

while True:
    try:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        height,width = gray.shape()
        ul = int(width/2-56),int(height/2-56)
        br = int(width/2+56),int(height/2+56)

        cv2.rectangle(gray,ul,br,(0,255,0),2)

        roi = gray[ul[1]:br[1],ul[0]:br[0]]

        ip = Image.fromarray(roi)

        img = ip.convert("L")
        imgresize = img.resize((28,28),Image.ANTIALIAS)

        imageinvert = PIL.ImageOps.invert(imgresize)

        pixelfilter = 20

        minpixel = np.percentile(imageinvert,pixelfilter)

        imginvertscale = np.clip(imageinvert-minpixel,0,255)

        maxpixel = np.max(imageinvert)

        imginvertscale = np.asarray(imageinvertscale/maxpixel,0,255)

        testSample = np.array(imginvertscale).reshape(1,784)

        test_pred = clf.predict(testSample)

        print("The predicted class is: ",test_pred)

        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        pass

cv2.release()
cv2.destroyAllWindows()

3