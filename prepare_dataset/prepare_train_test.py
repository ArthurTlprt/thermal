import pandas as pd
import numpy as np
from PIL import Image


df = pd.read_csv("dataset.csv")


train=df.sample(frac=0.8,random_state=200)
test=df.drop(train.index)

train.to_csv("train.csv")
test.to_csv("test.csv")
