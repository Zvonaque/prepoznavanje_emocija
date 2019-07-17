# učitavanje biblioteka i argumenata
import numpy as np
import matplotlib.pyplot as plt
import sys

py_filename = sys.argv[0]
classes = sys.argv[1]
conf_mat = sys.argv[2]


# definiranje funkcije za grafički prikaz matrice zabune
def plot_confusion_matrix(normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):

    # unos podataka matrice zabune
    cm = np.array([conf_mat])
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalizirana matrica zabune")
    else:
        print('Matrica zabune bez normalizacije')

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=[-0.5, 0, 1, 2, 3, 4, 5, 6, 6.5],
           yticks=[-0.5, 0, 1, 2, 3, 4, 5, 6, 6.5],
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Stvarna oznaka',
           xlabel='Predviđena oznaka')

    plt.setp(ax.get_xticklabels(), rotation=45,
             ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh
                    else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)
plot_confusion_matrix(title='Matrica zabune bez normalizacije')
plot_confusion_matrix(normalize=True,
                      title='Normalizirana matrica zabune')
plt.show()
