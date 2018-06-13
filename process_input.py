import cv2
import numpy as np
from scipy import ndimage
import math

#function finds center of mass of image and returns shiftx and shifty accordingly
def getBestShift(img):
	cy,cx = ndimage.measurements.center_of_mass(img)

	rows,cols = img.shape
	shiftx = np.round(cols/2.0-cx).astype(int)
	shifty = np.round(rows/2.0-cy).astype(int)

	return shiftx,shifty

#shifts image such that center of mass of image is in center of image
def shift(img,sx,sy):
	rows,cols = img.shape
	M = np.float32([[1,0,sx],[0,1,sy]])
	shifted = cv2.warpAffine(img,M,(cols,rows))
	return shifted

#croping the image such that black boundary is removed
def preprocess(gray):
	#remove useless boundry
	while np.sum(gray[0]) ==0 and gray.shape[0]>10:
		gray = gray[1:]

	while np.sum(gray[:,0]) == 0 and gray.shape[1]>10:
		gray = np.delete(gray,0,1)
	
	while np.sum(gray[-1]) == 0 and gray.shape[0]>10:
		gray = gray[:-1]

	while np.sum(gray[:,-1]) == 0 and gray.shape[1]>10:
		gray = np.delete(gray,-1,1)

	rows,cols = gray.shape
	#resize cropped image to 20*20
	if rows > cols:
		factor = 20.0/rows
		rows = 20
		cols = int(round(cols*factor))
		# first cols than rows
		gray = cv2.resize(gray, (cols,rows))
	else:
		factor = 20.0/cols
		cols = 20
		rows = int(round(rows*factor))
		# first cols than rows
		gray = cv2.resize(gray, (cols, rows))

	colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
	rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
	gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')
	
	#shift image so that center of mass is at center
	shiftx,shifty = getBestShift(gray)
	shifted = shift(gray,shiftx,shifty)
	gray = shifted
	return gray