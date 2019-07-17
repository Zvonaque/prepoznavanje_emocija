# učitavanje biblioteka i argumenata
import os
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import dlib
import cv2
import sys

py_filename = sys.argv[0]
predictor_path = sys.argv[1]
save_path = sys.argv[2]
root_images = sys.argv[3]

# pozivanje detektora lica i klase FaceAligner za poravnavanje lica
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
fa = FaceAligner(predictor, desiredFaceWidth=256)

# prolaz kroz sve slike u svim mapama zadane putanje
for subdir, dirs, files in os.walk(root_images):
    for file in files:
        image = cv2.imread(os.path.join(subdir, file))
        image = imutils.resize(image, width=800)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 2)

        # poravnavanje za svako lice pronađeno na slici
        i = 1
        for rect in rects:
            (x, y, w, h) = rect_to_bb(rect)
            faceAligned = fa.align(image, gray, rect)

            cv2.imwrite(save_path + file, faceAligned)
            cv2.waitKey(0)
            i += 1
