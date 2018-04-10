import numpy as np
import matplotlib.pyplot as plt

from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim

from keras.models import load_model
from data_loader import DataLoader

from skimage.measure import compare_ssim as ssim

generator = load_model("models/20.7344712317.h5")

def mse(x, y):
    return np.linalg.norm(x - y)


accuracy_ssim = []
error_mse = []

img_rows = 128
img_cols = 128

dataset_name = 'thermal2rgb'
data_loader = DataLoader(dataset_name=dataset_name, img_res=(img_rows, img_cols))

imgs_A, imgs_B = data_loader.load_data(batch_size=2, is_testing=True)
fakes_A = generator.predict(imgs_B)
fakes_A = np.asarray(fakes_A,dtype=np.float64)

for i, img_A in enumerate(imgs_A):
    print(fakes_A[i].shape)
    print(img_A.shape)
    print(fakes_A[i].dtype)

    accuracy_ssim.append(ssim(fakes_A[i], img_A, data_range=img_A.max() - img_A.min(), multichannel=True))
    print(accuracy_ssim[i])

    error_mse.append(mse(img_A,fakes_A[i]))
    print(error_mse[i])

    plt.imshow(0.5 * fakes_A[i] + 0.5)
    plt.show()
    plt.imshow(0.5 * img_A + 0.5)
    plt.show()
