# učitavanje biblioteka i argumenata
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import time
import sys

py_filename = sys.argv[0]
images = sys.argv[1]
labels = sys.argv[2]

# inicijalizacija mjerenja vremena i nasumičnih brojeva
start_time = time.time()
np.random.seed(7)

# inicijalizacija oznaka
labels_ch = ['anger', 'neutral', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
labels_num = [0, 1, 2, 3, 4, 5, 6]

# kreiranje setova za trening, validaciju i test
train_images, test_images, train_labels, test_labels = train_test_split(np.load(images), np.load(labels),
                                                                        test_size=0.20, random_state=42)
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels,
                                                                      test_size=0.2, random_state=42)

X_train = np.array(train_images, dtype="f")[..., np.newaxis]
y_train = np.array(train_labels, dtype="f")
X_test = np.array(test_images, dtype="f")[..., np.newaxis]
y_test = np.array(test_labels, dtype="f")
X_val= np.array(val_images, dtype="f")[..., np.newaxis]
y_val = np.array(val_labels, dtype="f")

# kreiranje modela neuronske mreže
model = tf.keras.Sequential()
model.add(layers.Conv2D(16, input_shape=(256, 256, 1), kernel_size=3, padding="same"))
model.add(layers.Conv2D(16, kernel_size=3, padding="same"))
model.add(layers.MaxPool2D())
model.add(layers.Conv2D(32, kernel_size=3, padding="same"))
model.add(layers.Conv2D(32, kernel_size=3, padding="same"))
model.add(layers.MaxPool2D())
model.add(layers.Conv2D(64, kernel_size=3, padding="same"))
model.add(layers.Conv2D(64, kernel_size=3, padding="same"))
model.add(layers.MaxPool2D())
model.add(layers.Conv2D(128, kernel_size=3, padding="same"))
model.add(layers.Conv2D(128, kernel_size=3, padding="same"))
model.add(layers.MaxPool2D())
model.add(layers.Flatten())
model.add(layers.Dense(7, activation='sigmoid'))

# kompiliranje modela i spremanje povijesti učenja
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, batch_size=30, validation_data=(X_val, y_val))

# evaluacija modela
scores = model.evaluate(X_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

y_hat = model.predict(X_test, verbose=0)
y_hat = np.argmax(y_hat, axis=1)
y_test = np.argmax(y_test, axis=1)
print(classification_report(y_test, y_hat, target_names=labels_ch))

matrix = confusion_matrix(y_test, y_hat, labels=labels_num)
print(matrix)

# ispis i spremanje modela
model.summary()
model.save('cnn.model')

# ispis proteklog vremena za vrijeme treniranja modela
end_time = time.time()
print('Elapsed time: ', end_time-start_time)

# ispis povijesti učenja
print(history.history.keys())

# grafički prikaz točnosti modela za vrijeme treninga i testa
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Točnost modela')
plt.ylabel('Točnost')
plt.xlabel('Epoha')
plt.legend(['Trening', 'Test'], loc='upper left')
plt.grid()
plt.show()

# grafički prikaz gubitka modela za vrijeme treninga i testa
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Gubitak modela')
plt.ylabel('Gubitak')
plt.xlabel('Epoha')
plt.legend(['Trening', 'Test'], loc='upper left')
plt.grid()
plt.show()
