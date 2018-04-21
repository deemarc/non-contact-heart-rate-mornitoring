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
rotata all image in the specified folder 90 degree CCW
'''

import glob
from PIL import Image
import os
def imageRotataAll(path,degree):
    # get all the video from video_in folder
    fileReg = path + '/*.jpg'
    images = glob.glob(fileReg)
    for image in images:
        filename_ext = os.path.basename(image)
        img = Image.open(image)
        img = img.rotate(degree)
        img.save(image)

if __name__ == '__main__' :
    imageRotataAll('./img/deemarc_80',90)
        
