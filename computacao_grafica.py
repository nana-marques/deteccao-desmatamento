import cv2
import numpy as np

for i in range(1, 6):
    # Carrega as imagens de satélite
    imagem = cv2.imread(f'./img/desmatamento/desmatamento{i}.png')

    # Converte as imagens para escala de cinza
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # Aplica a segmentação por limiar
    _, imagem_binaria = cv2.threshold(imagem_cinza, 10, 255, cv2.THRESH_BINARY)

    # Define os intervalos de cores para as áreas de desmatamento
    menor_limite = np.array([80, 80, 80], dtype=np.uint8)
    maior_limite = np.array([120, 138, 145], dtype=np.uint8)

    # Aplica a segmentação baseada em cores
    mascara = cv2.inRange(imagem, menor_limite, maior_limite)

    # Combina as máscaras
    mascara_final = cv2.bitwise_and(imagem_binaria, mascara)

    # Encontra os contornos na máscara
    contornos, _ = cv2.findContours(mascara_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Desenha os contornos nas áreas de desmatamento
    resultado = imagem.copy()
    cv2.drawContours(resultado, contornos, -1, (0, 255, 255), thickness=cv2.FILLED)

    cv2.imwrite(f'./img/desmatamento/imagem_resultante{i}.jpg', resultado)