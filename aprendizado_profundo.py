import numpy as np
import matplotlib.pyplot as plt
import glob, os
import re
from PIL import Image
import keras
import tensorflow as tf

def pre_processamento(diretorio, tamanho_maximo):
    # Converte as imagens em matrizes e escala de cinza
    img = Image.open(diretorio).convert('L')

    # Deixas a proporção das imagens em 1:1
    LARGURA, ALTURA = img.size
    if LARGURA != ALTURA:
            m_min_d = min(LARGURA, ALTURA)
            img = img.crop((0, 0, m_min_d, m_min_d))

    # Dimensiona as imagens em um tamanho padrão menor do que a resolução da imagem real
    img.thumbnail(tamanho_maximo, Image.Resampling.LANCZOS)

    return np.asarray(img)

def conjunto_dados(diretorio, tamanho_maximo):
        imagens = []
        nomes = []

        os.chdir(diretorio)
        for arquivo in glob.glob("*.png"):
            img = pre_processamento(arquivo, tamanho_maximo)
            if re.match('desmatamento.*', arquivo):
                    imagens.append(img)
                    nomes.append(0)
            elif re.match('nao_desmatada.*', arquivo):
                    imagens.append(img)
                    nomes.append(1)

        return (np.asarray(imagens), np.asarray(nomes))

diretorio_atual = os.getcwd()
tamanho_maximo = 130, 130
# Conjunto de dados para treinamento
(imagens_treinamento, nomes_treinamento) = conjunto_dados(f'{diretorio_atual}\\img\\conjunto_dados\\treinamento', tamanho_maximo)
# Conjunto de dados para testes
(imagens_testes, nomes_testes) = conjunto_dados(f'{diretorio_atual}\\img\\conjunto_dados\\testes', tamanho_maximo)

nomes = ['desmatamento', 'floresta']

# 16 imagens de treinamento
imagens_treinamento.shape
(50, 130, 130)
print(nomes_treinamento)

# 6 imagens de testes
imagens_testes.shape
(6, 130, 130)
print(nomes_testes)

# Classifica as imagens em valores entre 0 e 1
imagens_treinamento = imagens_treinamento / 255.0
imagens_testes = imagens_testes / 255.0

# Configura as camadas do modelo
modelo = keras.Sequential([
    keras.layers.Flatten(input_shape=(130, 130)), # Achata as imagens em um arranjo
    keras.layers.Dense(128, activation=tf.nn.sigmoid),
    keras.layers.Dense(16, activation=tf.nn.sigmoid),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

# Compila o modelo
sgd = keras.optimizers.SGD(lr=0.01, momentum=0.7, nesterov=True)
modelo.compile(optimizer=sgd,
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Treina o modelo
modelo.fit(imagens_treinamento, nomes_treinamento, epochs=6)

# Avalia a precisão
precisao_teste, acuracia_teste = modelo.evaluate(imagens_testes, nomes_testes)
print('Acurácia teste:', acuracia_teste)

# Previsões
previsoes = modelo.predict(imagens_testes)
print(previsoes)

def resultado(imagens, tipos):
        plt.figure(figsize=(10,10))
        tamanho = min(25, len(imagens))

        for i in range(tamanho):
            plt.subplot(5, 5, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(imagens[i], cmap=plt.cm.binary)
            plt.xlabel(nomes[tipos[i]])

# Mostra o resultado
resultado(imagens_testes, np.argmax(previsoes, axis = 1))
plt.show()