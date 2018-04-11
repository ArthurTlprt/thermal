import sys

import numpy as np
import matplotlib.pyplot as plt

from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim

from keras.models import load_model
from data_loader import DataLoader

from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt


n = int(sys.argv[1])
print(n)
generator = load_model("models/AB/5.38045740128.h5")

def mse(x, y):
    return np.linalg.norm(x - y)

def mean(l):
    return sum(l)/float(len(l))
    
accuracy_ssim_A = []
accuracy_ssim_B = []
error_mse_A = []
error_mse_B = []

img_rows = 128
img_cols = 128

dataset_name = 'thermal2rgb'
data_loader = DataLoader(dataset_name=dataset_name, img_res=(img_rows, img_cols))

for j in range(n//10):
    imgs_A, imgs_B = data_loader.load_data(batch_size=10, is_testing=True)
    fakes_A = generator.predict(imgs_B)
    fakes_A = np.asarray(fakes_A,dtype=np.float64)
    plt.imshow(imgs_A)
    plt.show()
    plt.imshow(fakes_A)

    for i, img_A in enumerate(imgs_A):
        ac = ssim(fakes_A[i], img_A, data_range=img_A.max() - img_A.min(), multichannel=True)
        accuracy_ssim_A.append(ac)

        err = mse(img_A,fakes_A[i])
        error_mse.append(err)



print("Thermique à RGB : ssim -"+mean(accuracy_ssim_A)+" mse -"+mean(error_mse_A))
