from PIL import Image
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
import os


A = [] # therm
B = [] # rgb

def reshape_file(path,file):
    global A, B
    f = open(path)
    f = list(f)
    for line in f:
        # ajout du path complet
        line = line[:-1] + ".jpg"
        A.append(line[:11]+"lwir/"+line[11:])
        B.append(line[:11]+"visible/"+line[11:])
    # pool = ThreadPool(8)
    # A = pool.map(reshape, A)
    # B = pool.map(reshape, B)

    for i,im in enumerate(A):
        #misc.imsave("trainA/"+str(i)+".png", im)
        im = reshape(im)
        im.save("datasets/thermal2rgb/%sA/"%file+str(i)+".jpg", "JPEG")
    for i,im in enumerate(B):
        #misc.imsave("trainB/"+str(i)+".png", im)
        im = reshape(im)
        im.save("datasets/thermal2rgb/%sB/"%file+str(i)+".jpg", "JPEG")


def reshape(path):
    #print(path)
    png_path = path[:-3]+"png"
    try:
        os.rename(path, png_path)
    except:
        pass
    im = Image.open(png_path)
    im = im.crop((64, 0, 576, 512))
    im = im.resize((128, 128))
    return im


reshape_file("test01.txt","test")
