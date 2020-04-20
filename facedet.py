import cv2 #package
load = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# used to load our required xml file

img = cv2.VideoCapture("sarthak.jpg")
# if you need to detect face on any camera use o or 1
# for any image pass the image name here

boolres,coordinatesofface = img.read()
# this read methid returns 2 param 1 is boolean that if image is read or not
# and second is the coordinates of the face on the image

grayimg = cv2.cvtColor(coordinatesofface,cv2.COLOR_BGR2GRAY)
#convert image in grayscale images
#because haarcascade_frontalface_ if for grayscale images

faces = load.detectMultiScale(grayimg,2,5)
#detecting faces takes 3 params
#1 grayimage 2 resizing command 3 neighbouring code
#returns x,y and height and width

for (x,y,width,height) in faces :
    cv2.rectangle(coordinatesofface,(x,y),(x+width,y+height),(255,255,0),2)
# rectangle takes 5 params
#1 image jisko scan krna
#2 x and y coordinates
# 3 top right side od rectanggle  baaki 2 points apne aap bna lega
# 4 color coordinates of box around face
# 5 width of the border of recatnage formed around face


cv2.imshow("Face detection",coordinatesofface)
#prints image takes 2 params title of the screen opened and coordinatesofface

# these 3 below are to open the image
cv2.waitKey(0)
#pass the time you want to keep the screen open if you want it to be infinite until you close it pass 0
# time taken in milli seconds

img.release()
#release the captured image

cv2.destroyAllWindows()
#closes the window
