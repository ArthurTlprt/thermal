import numpy as np
import matplotlib.pyplot as plt

from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim

from keras.models import load_model
from data_loader import DataLoader

from skimage.measure import compare_ssim as ssim

generator = load_model("models/13.1069565564inv.h5")

img_rows = 256
img_cols = 256
dataset_name = 'thermal2rgb'
data_loader = DataLoader(dataset_name=dataset_name, img_res=(img_rows, img_cols))

imgs_A, imgs_B = data_loader.load_data(batch_size=2, is_testing=True)
fake_A = generator.predict(imgs_B)


fake_A = np.resize(fake_A,(128,128,3))
imgs_B = np.resize(imgs_B,(128,128,3))
fake_A = np.asarray(fake_A,dtype='float')
imgs_B = np.asarray(imgs_B,dtype='float')
m_A = mse(imgs_B,fake_A)
s_A = ssim(imgs_B, fake_A, multichannel=True)
print("MSE: "+str(m_A)+" SSIM: "+str(s_A))


print(fake_A.shape)

gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

# Rescale images 0 - 1
gen_imgs = 0.5 * gen_imgs + 0.5

img = plt.imshow(gen_imgs[0])
plt.show()
