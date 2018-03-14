import pandas as pd
import numpy as np
from PIL import Image



# getting a unique dataframe
main_data_set = pd.read_csv("../trimodal-people-segmentation/TrimodalDataset/Scene 1/annotations.csv"
    , delimiter=";")[["Thermal file", "Thermal mask file"]]
main_data_set["Thermal file"]  = "Scene 1/" + main_data_set["Thermal file"]
main_data_set["Thermal mask file"]  = "Scene 1" + main_data_set["Thermal mask file"].str[1:]


data_set_2 = pd.read_csv("../trimodal-people-segmentation/TrimodalDataset/Scene 2/annotations.csv"
    , delimiter=";")[["Thermal file", "Mask file"]]
data_set_2["Thermal file"] = "Scene 2/" + data_set_2["Thermal file"].str[:-4] + ".jpg"
data_set_2["Thermal mask file"] = "Scene 2/thermalMasks/" + data_set_2["Mask file"]
data_set_2 = data_set_2.drop(["Mask file"], axis=1)

data_set_3 = pd.read_csv("../trimodal-people-segmentation/TrimodalDataset/Scene 3/annotations.csv"
    , delimiter=";")[["Thermal file", "Thermal mask file"]]
data_set_3["Thermal file"]  = "Scene 3/" + data_set_3["Thermal file"]
data_set_3["Thermal mask file"]  = "Scene 3" + data_set_3["Thermal mask file"].str[1:]


frames = [main_data_set, data_set_2, data_set_3]

main_data_set = pd.concat(frames)


# converting png in jpg

def convert_to_png(local_path):
    path_to_local = "../trimodal-people-segmentation/TrimodalDataset/"
    try:
        im = Image.open(path_to_local + local_path)
        try:
            rgb_im = im.convert('RGB')
            rgb_im.save(path_to_local + local_path[:-4] + ".jpg")
        except:
            print("unable to convert:")
            print(path_to_local+local_paths)
    except:
        print("unable to open:")
        print(path_to_local+local_path)


# for index, row in main_data_set.iterrows():
#     convert_to_png(row["Thermal file"])
#     convert_to_png(row["Thermal mask file"])

# save csv

for index, row in main_data_set.iterrows():
    row["Thermal file"] = row["Thermal file"][:-4] + ".jpg"
    row["Thermal mask file"] = row["Thermal mask file"][:-4] + ".jpg"

print(main_data_set)
# main_data_set.to_csv("dataset.csv")
