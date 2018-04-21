__copyright__ = """

    Copyright [2018] [MR.Deemarc Burakitbumrung]

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

"""
__license__ = "Apache 2.0"

'''
converting video into image sequence
'''

import cv2
import numpy as np
import os
import glob
# Find OpenCV version
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

# get all the video from video_in folder
videos = glob.glob('./video_in/*.mp4')

try:
    if not os.path.exists('img'):
        os.makedirs('img')
except OSError:
    print ('Error: Creating directory of data')

for video in videos:
    filename_ext = os.path.basename(video)
    filename, file_extension  = os.path.splitext(filename_ext)
    output_folder = './img/' + filename
    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    except OSError:
        print ('Error: Creating directory of data')

    # get video handle
    cap = cv2.VideoCapture(video)
    if int(major_ver)  < 3 :
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else :
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    #for now let's use just 1 second of video
    second = 1
    num_img = int(round(fps*second))
    print("number of img: " + str(num_img))

    frame_num = 0
    while(True):
        ret, frame = cap.read()
        if not ret: break
        output_filename = output_folder +'/'+ filename + '_' + ('%04d' %frame_num)+'.jpg'
        print ('Creating...' + output_filename)
        cv2.imwrite(output_filename, frame)
        frame_num += 1     

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()