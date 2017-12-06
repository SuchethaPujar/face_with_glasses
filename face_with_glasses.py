import cv2
import numpy as np


#img = cv2.imread('IMG_20171205_163456.jpg')
#img = cv2.imread('IMG_20171205_165615.jpg')
#img = cv2.imread('IMG_20171205_171122.jpg')
img=cv2.imread('image.jpg')

RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect the face in the image
haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
haar_eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=8);
#eyes=haar_eye_cascade.detectMultiScale(gray_img)
count=0
# Loop in all detected faces - in our case it is only one
#for i in range(1,5):
        #for (x,y,w,h) in faces:
for (i, (x, y, w, h)) in enumerate(faces):
#for (i, (x, y, w, h)) in faces:
                cv2.rectangle(RGB_img,(x,y),(x+w,y+h),(255,0,0), 5)
                cv2.putText(RGB_img, "Face_glass #{}".format(i + 1), (x, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                count=count+1

print("total:", count)
cv2.imshow("Faces",RGB_img)
cv2.waitKey(0)
