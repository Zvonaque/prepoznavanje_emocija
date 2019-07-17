# učitavanje biblioteka i argumenata
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import dlib
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys

py_filename = sys.argv[0]
classes = sys.argv[1]
image = sys.argv[2]
haarcascade_path = sys.argv[3]
predictor_path = sys.argv[4]
np.random.seed(7)

# algoritam za poravnavanje lica
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
fa = FaceAligner(predictor, desiredFaceWidth=256)

image = cv2.imread(image)
image = imutils.resize(image, width=800)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rect = detector(gray, 2)
(x, y, w, h) = rect_to_bb(rect)
faceAligned = fa.align(image, gray, rect)

# predikcija modela
model = tf.keras.models.load_model('cnn.model')
prediction = model.predict(faceAligned)
result = classes[np.where(prediction[0] == max(prediction[0]))[0][0]]

# prikaz slike s oznakom emocije
original_image = cv2.imread(image)
grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier(haarcascade_path)
detected_faces = face_cascade.detectMultiScale(grayscale_image)

for (column, row, width, height) in detected_faces:
    cv2.rectangle(
        original_image,
        (column, row),
        (column + width, row + height),
        (0, 0, 255),
        2
    )
    cv2.putText(original_image, result, (column - 2, row-5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

cv2.imshow('Image', original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# prikaz rezultata predikcije
plt.plot(classes,
         [prediction[0][0]*100, prediction[0][1]*100, prediction[0][2]*100, prediction[0][3]*100,
          prediction[0][4]*100, prediction[0][5]*100, prediction[0][6]*100])
plt.xlabel('Emocija')
plt.ylabel('Predikcija')
plt.title('Predikcija emocije')
plt.grid()
plt.show()
