import sys

import numpy as np
import matplotlib.pyplot as plt

from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim

from keras.models import load_model
from data_loader import DataLoader
from keras_contrib.layers.normalization import InstanceNormalization


from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt


n = int(sys.argv[1])
print(n)
generator = load_model("models/AB/3.61338320374.h5")

def mse(x, y):
    return np.linalg.norm(x - y)

def mean(l):
    return sum(l)/float(len(l))

accuracy_ssim = []
error_mse = []

img_rows = 128
img_cols = 128

dataset_name = 'thermal2rgb'
data_loader = DataLoader(dataset_name=dataset_name, img_res=(img_rows, img_cols))

for j in range(n//10):
    imgs_A, imgs_B = data_loader.load_test_data(batch_size=10)
    fakes_A = generator.predict(imgs_B)
    fakes_A = np.asarray(fakes_A,dtype=np.float64)


    for i, img_A in enumerate(imgs_A):
        plt.show()
        plt.imshow(img_A)
        ac = ssim(fakes_A[i], img_A, data_range=img_A.max() - img_A.min(), multichannel=True)
        accuracy_ssim.append(ac)

        err = mse(img_A,fakes_A[i])
        error_mse.append(err)

print(mean(accuracy_ssim))
print(mean(error_mse))
