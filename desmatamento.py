import numpy as np
import matplotlib.pyplot as plt
import glob, os
import re
from PIL import Image
import keras
import tensorflow as tf

def jpeg_to_8_bit_greyscale(path, maxsize):
    img = Image.open(path).convert('L')   # convert image to 8-bit grayscale

    # Make aspect ratio as 1:1, by applying image crop.
    # Please note, croping works for this data set, but in general one
    # needs to locate the subject and then crop or scale accordingly.

    WIDTH, HEIGHT = img.size
    if WIDTH != HEIGHT:
            m_min_d = min(WIDTH, HEIGHT)
            img = img.crop((0, 0, m_min_d, m_min_d))

    # Scale the image to the requested maxsize by Anti-alias sampling.
    img.thumbnail(maxsize, Image.Resampling.LANCZOS)

    return np.asarray(img)

def load_image_dataset(path_dir, maxsize):
        images = []
        labels = []

        os.chdir(path_dir)
        for file in glob.glob("*.png"):
            img = jpeg_to_8_bit_greyscale(file, maxsize)
            if re.match('desmatamento.*', file):
                    images.append(img)
                    labels.append(0)
            elif re.match('nao_desmatada.*', file):
                    images.append(img)
                    labels.append(1)

        return (np.asarray(images), np.asarray(labels))

maxsize = 130, 130
(train_images, train_labels) = load_image_dataset('./img/conjunto_dados/treinamento', maxsize)

(test_images, test_labels) = load_image_dataset('C:/Users/OI416936/github/deteccao-desmatamento/img/conjunto_dados/testes', maxsize)

class_names = ['desmatamento', 'floresta']

train_images.shape
(16, 130, 130)
print(train_labels)

test_images.shape
(6, 130, 130)
print(test_labels)

# Pré-processamento dos dados

# Classifica as imagens em valores entre 0 e 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Configura as camadas
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(130, 130)), # Achata as imagens em um arranjo
    keras.layers.Dense(128, activation=tf.nn.sigmoid),
    keras.layers.Dense(16, activation=tf.nn.sigmoid),
    keras.layers.Dense(2, activation=tf.nn.softmax)

])

# Compila o modelo
sgd = keras.optimizers.SGD(lr=0.01, momentum=0.7, nesterov=True)
model.compile(optimizer=sgd,
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Treina o modelo
model.fit(train_images, train_labels, epochs=100)

# Avalia a precisão
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# Previsões
predictions = model.predict(test_images)
print(predictions)

def display_images(images, labels):
        plt.figure(figsize=(10,10))
        grid_size = min(25, len(images))

        for i in range(grid_size):
                plt.subplot(5, 5, i+1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(images[i], cmap=plt.cm.binary)
                plt.xlabel(class_names[labels[i]])

# Mostra o resultado
display_images(test_images, np.argmax(predictions, axis = 1))
plt.show()