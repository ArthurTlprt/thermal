import scipy
from glob import glob
import numpy as np

class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, domain, batch_size=1, is_testing=False):
        data_type = "train%s" % domain if not is_testing else "val%s" % domain
        path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))

        batch_images = np.random.choice(path, size=batch_size)

        imgs = []
        for img_path in batch_images:
            img = self.imread(img_path)
            if not is_testing:
                img = scipy.misc.imresize(img, self.img_res)

                if np.random.random() > 0.5:
                    img = np.fliplr(img)
            else:
                img = scipy.misc.imresize(img, self.img_res)
            imgs.append(img)

        imgs = np.array(imgs)/127.5 - 1.

        return imgs

    def load_test_data(self, batch_size=1):
        imgs_A = imgs_B = []
        for i in range(batch_size):
            path_B = glob('./datasets/%s/testB/*' % (self.dataset_name))
            img_B_path = np.random.choice(path_B)

            fname = img_B_path.split('/')[-1]
            img_A_path = './datasets/%s/testA/%s' % (self.dataset_name, fname)


            img_A = self.imread(img_A_path)
            img_A = [scipy.misc.imresize(img_A, self.img_res)]
            img_A = np.array(img_A)/127.5 - 1.

            img_B = self.imread(img_B_path)
            img_B = [scipy.misc.imresize(img_B, self.img_res)]
            img_B = np.array(img_B)/127.5 - 1.
            if batch_size == 1:
                return img_A, img_B
            imgs_A.append(img_A)
            imgs_B.append(img_B)
        return imgs_A, imgs_B

    def load_img(self, path):
        img = self.imread(path)
        img = scipy.misc.imresize(img, self.img_res)
        img = img/127.5 - 1.
        return img[np.newaxis, :, :, :]

    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
