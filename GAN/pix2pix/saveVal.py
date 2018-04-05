from PIL import Image
import numpy as np
import os

def copy():
	size = 41640

	for i in range(size):
		A = Image.open("../thermal2rgb/datasets/thermal2rgb/testA/"+str(i)+".jpg")
		A.load()
		B = Image.open("../thermal2rgb/datasets/thermal2rgb/testB/"+str(i)+".jpg")
		B.load()
		dataA = np.array(A)
		dataB = np.array(B)
		arrayIm = np.concatenate((dataB,dataA),1)
		im = Image.fromarray(arrayIm)
		if i>size/2:
			im.save("datasets/thermal2rgb/val/"+str(i)+".jpg", "JPEG")
		elif i<size/2:
			im.save("datasets/thermal2rgb/test/"+str(i)+".jpg", "JPEG")


def train():
	size = 50184

	for i in range(size):
		A = Image.open("../thermal2rgb/datasets/thermal2rgb/trainA/"+str(i)+".jpg")
		A.load()
		B = Image.open("../thermal2rgb/datasets/thermal2rgb/trainB/"+str(i)+".jpg")
		B.load()
		dataA = np.array(A)
		dataB = np.array(B)
		arrayIm = np.concatenate((dataB,dataA),1)
		im = Image.fromarray(arrayIm)
		im.save("datasets/thermal2rgb/train/"+str(i)+".jpg", "JPEG")


copy()
#train()
