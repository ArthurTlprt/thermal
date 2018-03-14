from mask_rcnn import *

path = "../trimodal-people-segmentation/TrimodalDataset/"
test_df = pd.read_csv(path+"test.csv")


def predict_mask(image):
    # Run detection
    results = model.detect([image], verbose=1)
    r = results[0]
    print("results",np.shape(results[0]),type(results[0]))
    person_idx = np.where(r['class_ids'] == 1)[0]
    visualize.display_instances(image,r['rois'][person_idx], r['masks'], r['class_ids'],class_names, r['scores'])
    # recupere les idx correspondant à un humain
    print(r['rois'][person_idx])
    # r['masks'][:,:,person_idx] = r['masks'][:,:,person_idx] * 255
    # #unique, counts = np.unique(r['masks'][:,:,person_idx], return_counts=True)
    # return r['masks'][:,:,person_idx]




predict_mask(skimage.io.imread(path+"Scene 1/SyncT/00108.jpg"))

# for i, row in test_df.iterrows():
#     print(row["Thermal mask file"])
#     predict_mask(skimage.io.imread(path+row["Thermal file"]))
