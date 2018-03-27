from mask_rcnn import *

path = "../trimodal-people-segmentation/TrimodalDataset/"
test_df = pd.read_csv(path+"test.csv")[:1]


def predict_box(image):
    # Run detection
    results = model.detect([image], verbose=1)
    r = results[0]
    print("results",np.shape(results[0]),type(results[0]))
    person_idx = np.where(r['class_ids'] == 1)[0]
    # visualize.display_instances(image,r['rois'][person_idx], r['masks'], r['class_ids'],class_names, r['scores'])
    # recupere les idx correspondant à un humain
    return r['rois'][person_idx]


test_df["boxes"] = np.nan
print(test_df)

for i, row in test_df.iterrows():
    print(row["Thermal file"])
    test_df["boxes",i] = str(predict_box(skimage.io.imread(path+row["Thermal file"])))

test_df.to_csv("box")
