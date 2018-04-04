
import csv
import numpy as np
from PIL import Image, ImageDraw, ImageOps

fichier = open("testi.csv", "a")
with open('train.csv') as f:
    has_header = csv.Sniffer().has_header(f.read(1024))
    f.seek(0)  # rewind
    incsv = csv.reader(f)
    if has_header:
        next(incsv)  # skip header row
    reader = csv.reader(f)
    for row in reader:
        #print(row)
        tab2=row[3].split('/')
        tab2[1]="SyncRGB"
        tab2=tab2[0]+"/SyncRGB/"+tab2[2]
        row=row[0]+","+row[1]+","+tab2+","+row[3]
        fichier.write(row+ "\n")


fichier.close()
