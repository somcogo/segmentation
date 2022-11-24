import nibabel as nib
import os
import matplotlib.pyplot as plt
import csv
import json

with open('./data/verse19/dataset-verse19test/derivatives/sub-verse032/sub-verse032_seg-vb_ctd.json') as f:
    data = json.load(f)
    for x in data[1:-1]:
        print(x['label'])
