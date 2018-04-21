import numpy as np
import cv2
import glob
import os

class person():
    def __init__(self):
        self.face = []
        self.eyes = []
# def draw_rectangle(img, pts):
#     """ 
#     desp: it going to draw a rectangle from pts given
#     input: pts = (x,y,w,h)
#     """

def detect(grayimg_src):
    """
    input: grayscale image,  
    ouput: people contain faces, eyes region of input image  s
    format of the output: (x,y,w,h)
    """
    # multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

    #https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
    face_cascade = cv2.CascadeClassifier('./detect_face/haarcascade_frontalface_default.xml')
    #https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
    eye_cascade = cv2.CascadeClassifier('./detect_face/haarcascade_eye.xml')
    faces = face_cascade.detectMultiScale(grayimg_src, 1.3, 5)
    people = []
    for (x,y,w,h) in faces:
        people.append(person())
        people[-1].face = (x,y,w,h)
        # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = grayimg_src[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for eye in eyes:
            people[-1].eyes.append(eye)

    return people



if __name__ == '__main__':
    imagesPath = glob.glob('./detect_face/*.jpg')
    i = 0
    for imagePath in imagesPath:
        # note that opencv give out image in bgr format
        image = cv2.imread(imagePath)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        people = detect(gray_img)
        
        for person in people:
            # draw rectangle on all the face
            x = person.face[0]
            y = person.face[1]
            w = person.face[2]
            h = person.face[3]
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)

            # draw rectangle on all the eyes of this face
            roi_color = image[y:y+h, x:x+w]
            eyes = person.eyes
            for (ex,ey,ew,eh) in eyes:
                 cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                 
        cv2.imshow('img',image)  
        print("person " + str(i))
        cv2.waitKey()




