from scipy import misc
from PIL import Image
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
import os


A = [] # therm
B = [] # rgb

def reshape_file(path):
    global A, B
    f = open(path)
    f = list(f)[:10]
    for line in f:
        # ajout du path complet
        line = line[:-1] + ".jpg"
        A.append(line[:11]+"lwir/"+line[11:])
        B.append(line[:11]+"visible/"+line[11:])
    pool = ThreadPool(4)
    A = pool.map(reshape, A)
    B = pool.map(reshape, B)

    for i,im in enumerate(A):
        #misc.imsave("trainA/"+str(i)+".png", im)
        im.save("dataset/thermal2rgb/trainA/"+str(i)+".jpg", "JPEG")
    for i,im in enumerate(B):
        #misc.imsave("trainB/"+str(i)+".png", im)
        im.save("dataset/thermal2rgb/trainB/"+str(i)+".jpg", "JPEG")

def reshape(path):
    png_path = path[:-3]+"png"
    try:
        os.rename(path, png_path)
    except:
        pass
    #im = misc.imread(png_path, mode="RGB")
    # print(im[0].min())
    # im = im[:,64:-64]        # crop -> 512*512*3
    # im = np.resize(im, (128,128,3))

    im = Image.open(png_path)
    im = im.crop((64, 0, 576, 512))
    im = im.resize((128, 128))
    return im

reshape_file("train01.txt")
