from PIL import Image
import numpy as np
import os

def val(name,size):
    for i in range(size):
        load = Image.open("datasets/thermal2rgb/test"+str(name)+"/"+str(i)+".jpg")
        load.load()
        data = np.array(load)
        arrayIm = np.concatenate(data,1)
        im = Image.fromarray(arrayIm)
        if i>size/2 :
            os.remove("datasets/thermal2rgb/test"+str(name)+"/"+str(i)+".jpg")
            im.save("datasets/thermal2rgb/val"+str(name)+"/"+str(i)+".jpg", "JPEG")

val("A",79362)
val("B",41640)
