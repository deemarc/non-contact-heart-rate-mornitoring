import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob
from detect_face import haarcc_faceDetect
from detect_human_skin import skinDetector
from scipy.spatial import distance

faceDet_al = {'manual-r':0, 'manaul-coor':1 , 'haar':2}
# when choosing to detect face manually please congig 'face_cropRatio'
# it is a percentage ration that you want to remove from left top right down
face_cropRatio = [0.36, 0.4, 0.45, 0.10]
face_cropCoor = [240, 107, 158, 158]
x = 0
y = 0
w = 0
h = 0
def faceDetect(img_in, algor=faceDet_al['manual-r']):
    global x,y,w,h

    img_x,img_y = img_in.shape[:2]
    if algor == faceDet_al['manual-r']:
        x = face_cropRatio[0]*img_x
        y = face_cropRatio[1]*img_y
        w = img_x - face_cropRatio[2]*img_x
        h = img_y - face_cropRatio[3]*img_y
    if algor == faceDet_al['manaul-coor']:
        x = face_cropCoor[0]
        y = face_cropCoor[1]
        w = face_cropCoor[2]
        h = face_cropCoor[3]
        
    elif algor == faceDet_al['haar']:
        gray_img = cv2.cvtColor(img_in,cv2.COLOR_BGR2GRAY)
        people = haarcc_faceDetect.detect(gray_img)
        # if more than one person has been detected throw an error!
        if len(people) > 1:
            if (x+y+w+h) == 0:
                x = face_cropRatio[0]*img_x
                y = face_cropRatio[1]*img_y
                w = img_x - face_cropRatio[2]*img_x
                h = img_y - face_cropRatio[3]*img_y
            else:
                #choose the one near the last one
                preFace = [x,y,w,h]
                minDist = 100000
                minFace = []
                for person in people:
                    curFace = [person.face[0],person.face[1],person.face[2],person.face[3]]
                    dst = distance.euclidean(preFace,curFace)
                    if dst < minDist:
                        minDist = dst
                        minFace = curFace
                x = minFace[0]
                y = minFace[1]
                w = minFace[2]
                h = minFace[3]
            #raise ValueError('more than one person has been detected in an image!!!')
        elif len(people) == 1:
            person = people[0]
            x = person.face[0]
            y = person.face[1]
            w = person.face[2]
            h = person.face[3]
        elif (x+y+w+h) == 0:
            x = face_cropRatio[0]*img_x
            y = face_cropRatio[1]*img_y
            w = img_x - face_cropRatio[2]*img_x
            h = img_y - face_cropRatio[3]*img_y

    else:
        raise ValueError('faceDet_al value cannot be recognise')

    return [x,y,w,h]

def roi_draw(img_in, x,y,w,h):
    roi_draw_img = img_in
    cv2.rectangle(roi_draw_img,(x,y),(x+w,y+h),(0,0,255),2)
    return roi_draw_img

def roi_crop(img_in, x,y,w,h):
    roi_crop_img = img_in[y:y+h, x:x+w]
    return roi_crop_img

def colorSpaceCovert(img_in):
    #let's try lot's of color space
    hls_img = cv2.cvtColor(img_in, cv2.COLOR_BGR2HLS)
    hsv_img = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
    lab_img = cv2.cvtColor(img_in, cv2.COLOR_BGR2Lab)

    img_dict = {'bgr_img':img_in, 'hls_img':hls_img, 'hsv_img':hsv_img, 'lab_img':lab_img}
    # img_list.append(hls_img)
    # img_list.append(hsv_img)
    # img_list.append(lab_img)
    return img_dict


# def cal_mm(img_in, com3=True):
#     if com3:
#         n = 3
#     else:
#         n = 1
#     mean = []   
#     median = []
#     for i in range(0,n):
#         mean.append(np.mean(img_in[:,:,i]))
#         median.append(np.median(img_in[:,:,i]))
    
#     return mean, median

def cal_mm(img_in, skinMask, com3=True, enb_skinMask = True):

    if com3:
        n = 3
    else:
        n = 1

    mean = []   
    median = []
    for i in range(0,n):
        mean.append(np.mean(img_in[:,:,i]))
        median.append(np.median(img_in[:,:,i]))
    
    if enb_skinMask:
        # using skin mask as filter
        for i in range(0, n):
            cur_arr = img_in[:,:,i]
            fil_arr = cur_arr[skinMask==1]
            mean.append(np.mean(fil_arr))
            median.append(np.median(fil_arr))

    return mean, median

def kernel(img_in, seq_arr, cvRead=True):
    
    # face detection
    x,y,w,h = faceDetect(img_in,faceDet_al['manaul-coor'])

    # for now just draw the roi
    # roi_draw_img = roi_draw(img_in,x,y,w,h)
 
    skinMask = skinDetector.detect(img_in)
    roi_crop_img = roi_crop(img_in,x,y,w,h)
    roi_crop_skinMask = roi_crop(skinMask,x,y,w,h)
    
    pro_img = cv2.bitwise_and(img_in, img_in, mask = skinMask)
    roi_draw_img = roi_draw(pro_img,x,y,w,h)
    # print(x,y,w,h)
    # cv2.imshow("roi", roi_draw_img)
    # cv2.waitKey()
    # colorspacCoversion
    img_dict = colorSpaceCovert(roi_crop_img)
    # finding mean and median
    for item_name in img_dict:
        cur_img = img_dict[item_name]
        mean, median = cal_mm(cur_img,roi_crop_skinMask)
        img_dict[item_name] = [mean, median]
        
    seq_arr.append(img_dict)
    # put into time series array 
    return seq_arr,roi_draw_img  

def kernel_visualize(img_in):
    # face detection
    x,y,w,h = faceDetect(img_in)

#trying out kernel

if __name__ == '__main__':
    # get all image folder
    path = "./img/"
    dirs = os.listdir(path)
    image_paths = []
    for directory in dirs:
        image_path = path + directory + '/'
        image_paths.append(image_path)

    # for now get all image in fisrt folder
    img_arr = []
    fileReg = image_paths[0] + '*jpg'
    images = glob.glob(fileReg)
    img_arr.append(images)

    #testing out our kernel
    seq_arr = []
    #first 10 picture
    images = img_arr[0][2:4]
    for img in images:
        cur_img = cv2.imread(img)
        cv2.imshow("images", cur_img)
        cv2.waitKey()
        seq_arr = kernel(cur_img,seq_arr)