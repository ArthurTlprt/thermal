from mask_rcnn import *

path = "../trimodal-people-segmentation/TrimodalDataset/"


test_df = pd.read_csv(path+"test.csv")

def predict_mask(image):
    # Run detection
    results = model.detect([image], verbose=1)
    r = results[0]
    #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],class_names, r['scores'])
    # recupere les idx correspondant à un humain
    person_idx = np.where(r['class_ids'] == 1)[0]
    r['masks'][:,:,person_idx] = r['masks'][:,:,person_idx] * 255
    #unique, counts = np.unique(r['masks'][:,:,person_idx], return_counts=True)
    return r['masks'][:,:,person_idx]

def compute_accuracy(row):
    pred_mask = predict_mask(skimage.io.imread(path+row["Thermal file"]))
    real_mask = skimage.io.imread(path+row["Thermal mask file"])


    # j'additionne les images et compte les pixels > 255
    # si predit humain alors pixel = 255
    # si mask dit humain alors pixel > 0

    real_mask = np.sum(real_mask, axis=2, dtype=np.uint16)
    pred_mask = np.sum(pred_mask, axis=2, dtype=np.uint16)

    unique, counts = np.unique(real_mask, return_counts=True)
    occurence = dict(zip(unique, counts))

    n_pix_mask = 0
    for i in occurence:
        if i > 0:
            n_pix_mask += occurence[i]

    superposition = np.zeros((480,640), dtype=np.uint16)
    superposition = superposition + real_mask + pred_mask

    unique, counts = np.unique(superposition, return_counts=True)
    occurence = dict(zip(unique, counts))

    # je compte les pixels > 255
    n_superpositions = 0
    for i in occurence:
        if i > 255:
            n_superpositions += occurence[i]
    try:
        accu = n_superpositions/n_pix_mask
        print(accu)
    except:
        print("erreur, division par zéro")
        accu=0
    return accu

test_df['accuracy_mask'] = np.nan

for i, row in test_df.iterrows():
    print("######################################")
    print(row["Thermal file"])
    print(row["Thermal mask file"])
    print("")
    accu = compute_accuracy(row)
    test_df['accuracy_mask'][i] = accu
    if i % 10 == 0.:
        test_df.to_csv("../performances/tmp")
    print("######################################")

test_df.to_csv("../performances/" + str(test_df['accuracy_mask'].mean()))

print("it works")
