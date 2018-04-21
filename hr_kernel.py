import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob
from detect_face import haarcc_faceDetect
from detect_human_skin import skinDetector

# cropRatio = [x_left,y_top,x_right,y_bottom]
def roi_faceDetect(img_in):
    gray_img = cv2.cvtColor(img_in,cv2.COLOR_BGR2GRAY)
    people = haarcc_faceDetect.detect(gray_img)
    # if more than one person has been detected throw an error!
    if len(people) > 1:
        raise ValueError('more than one person has been detected in an image!!!')
    
    person = people[0]
    x = person.face[0]
    y = person.face[1]
    w = person.face[2]
    h = person.face[3]
    roi_img = img_in[y:y+h, x:x+w]
    return roi_img
    
def roi_manual(img_in, cropRatio = [0.36, 0.4, 0.45, 0.10], cvRead=True):
    h,w = img_in.shape[:2]

    x_left = cropRatio[0]*w
    y_top = cropRatio[1]*h

    x_right = w - cropRatio[2]*w
    y_bottom = h - cropRatio[3]*h

    if cvRead:
        img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)

    roi_img = img_in[int(y_top):int(y_bottom), int(x_left):int(x_right), :]
    return roi_img

def colorSpaceCovert(img_in):
    #let's try lot's of color space
    hls_img = cv2.cvtColor(img_in, cv2.COLOR_RGB2HLS)
    hsv_img = cv2.cvtColor(img_in, cv2.COLOR_RGB2HSV)
    lab_img = cv2.cvtColor(img_in, cv2.COLOR_RGB2Lab)

    img_dict = {'rgb_img':img_in, 'hls_img':hls_img, 'hsv_img':hsv_img, 'lab_img':lab_img}
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
    # finding roi
    # roi_img = roi_manual(img_in)
    roi_img = roi_faceDetect(img_in)
    skinMask = skinDetector.detect(roi_img)
    # colorspacCoversion
    img_dict = colorSpaceCovert(roi_img)
    # finding mean and median
    for item_name in img_dict:
        cur_img = img_dict[item_name]
        mean, median = cal_mm(cur_img,skinMask)
        img_dict[item_name] = [mean, median]
        
    seq_arr.append(img_dict)
    # put into time series array 
    return seq_arr  


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
    images = img_arr[0][0:10]
    for img in images:
        cur_img = cv2.imread(img)
        seq_arr = kernel(cur_img,seq_arr)