import cv2 as cv

img=cv.imread("steve.jpg")
resized_img=cv.resize(img,(500,500),interpolation=cv.INTER_AREA)
#cv.imshow("Captain America",resized_img)

gray=cv.cvtColor(resized_img,cv.COLOR_BGR2GRAY)
#cv.imshow("Grayscale",gray)

classifier= cv.CascadeClassifier("haar_face.xml")

rect=classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)
print("No of faces=",len(rect))

for (x,y,w,h) in rect:
    cv.rectangle(resized_img,(x,y),(x+w,y+h),(0,0,255),thickness=2)
cv.imshow("Faces detected",resized_img)
cv.waitKey(0)
