# učitavanje biblioteka i argumenata
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
from skimage import io
import sys
import os
import random

py_filename = sys.argv[0]
folder_path = sys.argv[1]
new_folder = sys.argv[2]


# definiranje funkcije za nasumičnu rotaciju
def random_rotation(image_array: ndarray, deg):
    random_degree = random.uniform(-deg, deg)
    return sk.transform.rotate(image_array, random_degree)


# definiranje funkcije za nasumični šum
def random_noise(image_array: ndarray):
    return sk.util.random_noise(image_array)


# definiranje funckije za horizontalno zrcaljenje
def horizontal_flip(image_array: ndarray):
    return image_array[:, ::-1]


# prolaz kroz sve slike u svim mapama zadane putanje
for subdir, dirs, files in os.walk(folder_path):
    for file in files:
        image_path = os.path.join(subdir, file)
        image_to_transform = sk.io.imread(image_path)

        # nasumične rotacije
        rotations = [15, 15, 25, 25, 45, 45]
        for i in range(0, 6):
            rotated_image = random_rotation(image_to_transform, i)
            io.imsave(new_folder + file.split('.')[0] +
                      '_rotate_' + str(i + 1) + '.png', rotated_image)

        # nasumični šum
        for i in range(0, 3):
            noisy_image = random_noise(image_to_transform)
            io.imsave(new_folder + file.split('.')[0] +
                      '_noise_' + str(i + 1) + '.png', noisy_image)

        # horizontalno zrcaljenje
        flipped_image = horizontal_flip(image_to_transform)
        io.imsave(new_folder + file.split('.')[0] + '_flip'
                  + '.png', flipped_image)
