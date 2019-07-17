# učitavanje biblioteka i argumenata
import numpy as np
import cv2
import sys
import os

py_filename = sys.argv[0]
folder_path = sys.argv[1]
folder_path_2 = sys.argv[1]

# prolaz kroz sve slike u svim mapama zadane putanje
loaded_images = []
for subdir, dirs, files in os.walk(folder_path):
    for file in files:
        img_data = cv2.imread(os.path.join(subdir, file), cv2.IMREAD_GRAYSCALE)
        loaded_images.append(img_data)

# spremanje slika u obliku .npy datoteke
np.save('loaded_images' + '.npy', loaded_images)


# definiranje funkcije za kreaciju oznaka slika
def lab_arr(x):
    return {
        '1': [1, 0, 0, 0, 0, 0, 0],
        '2': [0, 1, 0, 0, 0, 0, 0],
        '3': [0, 0, 1, 0, 0, 0, 0],
        '4': [0, 0, 0, 1, 0, 0, 0],
        '5': [0, 0, 0, 0, 1, 0, 0],
        '6': [0, 0, 0, 0, 0, 1, 0],
        '7': [0, 0, 0, 0, 0, 0, 1]
    }[x]


# prolaz kroz sve slike u svim mapama zadane putanje
loaded_labels = []
for subdir, dirs, files in os.walk(folder_path_2):
    for file in files:
        f = open(os.path.join(subdir, file), "r")
        label = f.read()[3:4]
        f.close()

        # sveukupno 15 slika za svaku originalnu sliku
        for i in range(0, 15):
            loaded_labels.append(lab_arr(label))

# spremanje oznaka slika u obliku .npy datoteke
np.save('loaded_labels' + '.npy', loaded_labels)
