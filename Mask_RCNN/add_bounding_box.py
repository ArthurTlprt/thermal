from mask_rcnn import *

path = "../trimodal-people-segmentation/TrimodalDataset/"
test_df = pd.read_csv(path+"test.csv")[:2]


def predict_box(image):
    # Run detection
    results = model.detect([image], verbose=1)
    r = results[0]
    print("results",np.shape(results[0]),type(results[0]))
    person_idx = np.where(r['class_ids'] == 1)[0]
    # visualize.display_instances(image,r['rois'][person_idx], r['masks'], r['class_ids'],class_names, r['scores'])
    # recupere les idx correspondant à un humain
    return r['rois'][person_idx]


test_df["boxes"] = ""

for i, row in test_df.iterrows():
    print("detecting box of file "+str(i)+"/1591")
    print(row["Thermal file"])
    print(str(predict_box(skimage.io.imread(path+row["Thermal file"]))))
    box = str(predict_box(skimage.io.imread(path+row["Thermal file"])))
    test_df.set_value(i, "boxes", box)
    if i % 10 == 0.:
        test_df.to_csv("test_box_temp")

test_df = test_df.drop(["Unnamed: 0", "Unnamed: 0.1"], axis=1)
print(test_df)
test_df.to_csv("test_box.csv")
