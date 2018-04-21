# import the necessary packages
import numpy as np
import argparse
import cv2
import glob

# define the upper and lower boundaries of the HSV pixel
# intensities to be considered 'skin'
lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")

def detect(bgr_img):
	hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
	skinMask = cv2.inRange(hsv_img, lower, upper)
	# apply a series of erosions and dilations to the mask
	# using an elliptical kernel
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
	skinMask = cv2.erode(skinMask, kernel, iterations = 2)
	skinMask = cv2.dilate(skinMask, kernel, iterations = 1)


	# blur the mask to help remove noise, then apply the
	# mask to the frame
	skinMask = cv2.GaussianBlur(skinMask, (5, 5), 0)
	return skinMask

if __name__ == '__main__':
	imagesPath = glob.glob('./detect_human_skin/*.jpg')

	for imagePath in imagesPath:
		# note that opencv give out image in bgr format
		image = cv2.imread(imagePath)
		skinMask = detect(image)
		ret_img = cv2.bitwise_and(image, image, mask = skinMask)
		arr = []
		for x in range(0,ret_img.shape[0]):
			for y in range(0,ret_img.shape[1]):
				if skinMask[x,y] == 1:
					arr.append(ret_img[x,y,:])
		print(arr)
		# show the skin in the image along with the mask
		cv2.imshow("images", ret_img)
		cv2.waitKey()
		cv2.imshow("images2", arr)
		cv2.waitKey()
